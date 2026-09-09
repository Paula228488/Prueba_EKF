#!/usr/bin/env python3
"""
simulacion_batch.py
Versión "sin frontend" de simulacion_geometrica.py: motor de posicionamiento
geométrico puro (LM + EKF 6-DOF) reutilizado tal cual, pero conducido
automáticamente en DOS FASES:

  FASE 1 — BÚSQUEDA DE LA MEJOR COLOCACIÓN DE ANCLAS (sin ruido y con ruido)
    Para cada nº de anclas (3, 4, 5, 6) y cada sala de ROOMS se combina un
    criterio en dos pasos:
      1a. Sin ruido: se prueban tantas configuraciones de anclas como haga
          falta (layouts "semilla" conocidos + candidatos aleatorios sobre
          las paredes/techo/suelo de la sala) hasta reunir un POOL de
          candidatos cuyo error de estimación, con distancias EXACTAS
          (dist_noise = none), sea del orden de SEARCH_TARGET_ERROR (por
          defecto 1e-12, margen de seguridad sobre el ~1e-15 de doble
          precisión).
      1b. Con ruido: TODOS los candidatos del pool se evalúan bajo varios
          escenarios de error representativos (bias, gaussiano, uniforme y
          combinaciones heterogéneas por ancla + tag) y se elige el que dé
          MENOR error medio con ruido. Es decir, la búsqueda no se queda
          con el primer candidato "casi exacto sin ruido": analiza todas
          las posibilidades del pool bajo ruido para decidir la mejor
          configuración final.
    Las cuentas de anclas se recorren en orden ascendente dentro de cada
    sala: el objetivo es que a más anclas se consiga igualar o reducir
    tanto el error sin ruido como el error con ruido logrados con menos
    anclas (nunca empeorarlos, sin ruido NI con ruido).
    Cada intento/candidato es una simulación completa (EKF/LM) sobre
    varias trayectorias.
    Resultado: la MEJOR configuración encontrada para cada (n_anclas, sala),
    aunque no llegue exactamente a 1e-15 sin ruido (se registra el error
    alcanzado, sin ruido y con ruido).

  FASE 2 — BARRIDO DE ERRORES SOBRE LAS MEJORES CONFIGURACIONES
    Usando EXCLUSIVAMENTE las mejores configuraciones de la fase 1, se
    introducen los distintos tipos de error en la medida de distancia
    ancla-tag (bias/constante, gaussiano, uniforme) y sus combinaciones,
    incluyendo explícitamente la posibilidad de que el error añadido sea
    positivo o negativo (bias +/-, uniforme simétrico/solo-positivo/
    solo-negativo, gaussiano con media 0 o desplazada). También se prueban
    combinaciones de dos tipos de error a la vez (p. ej. bias + gaussiano).

Salida:
  outputs/<run>/busqueda_anclas_log.csv
      -> un registro por cada intento de la fase 1a (sin ruido): todas las
         configuraciones probadas y su error, para poder auditar la
         búsqueda.
  outputs/<run>/busqueda_anclas_log_ruido.csv
      -> un registro por cada candidato del pool evaluado en la fase 1b
         (con ruido): su error sin ruido, su error medio con ruido, y el
         mejor error con ruido logrado con menos anclas.
  outputs/<run>/mejores_configuraciones.csv
      -> una fila por (n_anclas, sala) con las coordenadas de la mejor
         configuración encontrada y el error alcanzado, sin ruido y con
         ruido.
  outputs/<run>/simulacion_uwb_<N>anclas_<sala>_<traj>_<escenario_error>.csv
      -> mismo formato que exportaba el botón "Descargar CSV" del frontend,
         compatible con comparador_global.py (secciones ANCLAS / DATOS POR
         PUNTO DE TRAYECTORIA / RESUMEN, columna 'Error_Estimacion(m)').
      -> solo se generan para las mejores configuraciones de la fase 1,
         incluyendo el escenario "sin_error" (baseline) y todo NOISE_SCENARIOS.
  outputs/<run>/resumen_global.csv
      -> una fila por combinación (n_anclas, sala, trayectoria, escenario de
         error, repetición) con error medio/RMS/máx, lista para graficar.

Uso:
    pip install numpy scipy
    python simulacion_batch.py
"""

import csv
import math
import os
import time
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

# ═══════════════════════════════════════════════════════════════════
# MOTOR DE POSICIONAMIENTO — copiado sin cambios de simulacion_geometrica.py
# (LM de trilateración + EKF 6-DOF). No se toca: es el algoritmo que ya
# estaba validado; aquí solo cambia cómo se generan y seleccionan los
# escenarios (anclas + error) que se le pasan.
# ═══════════════════════════════════════════════════════════════════

EKF_R_DIST = 1e-6
EKF_Q_POS = 1e-4
EKF_Q_VEL = 1e-3
EKF_MAX_DIST = 1000.0
EKF_MIN_DIST = 1e-4
EKF_WARMUP_SAMPLES = 6


def _plane_fit(points: np.ndarray):
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[2]
    return normal, centroid


def _solve_trilateration(anchors_used: list, interior_hint=None,
                          sigmas: Optional[list] = None) -> np.ndarray:
    """Resuelve la posición por mínimos cuadrados no lineales a partir de
    N medidas ancla-tag (N >= 3).

    Antes: residuos SIN ponderar y con pérdida cuadrática pura (method="lm").
    Esto es lo que provoca que "más anclas" pueda dar MÁS error que "menos
    anclas": con pérdida cuadrática y peso igual para todas las anclas, basta
    con que UNA sola de las anclas añadidas tenga, en esa muestra, una
    realización de ruido más grande de lo normal para que tire de la
    solución hacia sí con la misma fuerza que las demás — y cuantas más
    anclas hay, mayor es la probabilidad de que eso ocurra en al menos una
    de ellas. Con 3-4 anclas ese riesgo es bajo; con 5-6 crece.

    Ahora: (1) cada residuo se normaliza por la incertidumbre (sigma) de esa
    ancla si se conoce (mínimos cuadrados generalizados / ponderados, en vez
    de asumir que todas las anclas son igual de fiables), y (2) se usa una
    pérdida robusta (loss="soft_l1") en vez de cuadrática pura: un residuo de
    ~1 sigma pesa lo normal, pero uno de varias sigmas (outlier) se
    downweightea en vez de arrastrar la solución con toda su fuerza. Con
    esto, añadir anclas solo puede aportar información extra (correctamente
    ponderada) sin que una única medida mala domine el resultado — que es la
    propiedad que hace falta para que el error no empeore al pasar de 4 a 5
    o 6 anclas. method="trf" es necesario porque method="lm" de scipy no
    admite loss distinto de "linear"."""
    w = [1.0 / (d + 1e-6) for _, d in anchors_used]
    sw = sum(w)
    p0 = sum(wi * a for (a, _), wi in zip(anchors_used, w)) / sw

    anchor_pts = np.array([a for a, _ in anchors_used])
    if len(anchor_pts) >= 3:
        normal, centroid = _plane_fit(anchor_pts)
        spread = max(float(np.max(np.linalg.norm(anchor_pts - centroid, axis=1))), 1e-3)
        offset = 0.15 * spread

        if interior_hint is not None:
            side = np.sign(np.dot(np.asarray(interior_hint, dtype=float) - centroid, normal))
            if side == 0:
                side = 1.0
        else:
            side = 1.0

        p0 = p0 + side * offset * normal

    n = len(anchors_used)
    if sigmas is None:
        sig = np.ones(n)
    else:
        sig = np.maximum(np.asarray(sigmas, dtype=float), 1e-6)
        if sig.shape[0] != n:
            sig = np.ones(n)

    def residuals(p):
        return [(np.linalg.norm(p - a) - d) / s for (a, d), s in zip(anchors_used, sig)]

    try:
        res = least_squares(residuals, p0, method="trf", loss="soft_l1",
                             f_scale=1.0, max_nfev=400)
        return res.x
    except Exception:
        try:
            res = least_squares(residuals, p0, method="lm", max_nfev=400)
            return res.x
        except Exception:
            return p0


class PositionEKF:
    _CHI2_GATE = 10.83
    _INNOV_WINDOW = 20

    def __init__(self, anchor_positions: dict, exact_mode: bool = False):
        self.anchor_positions = anchor_positions
        self._initialized = False
        self._warmup_buf = []

        self.x = np.zeros(6)
        self.P = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])

        _q_scale = 1e-8 if exact_mode else 1.0
        self._q_pos = EKF_Q_POS * _q_scale
        self._q_vel = EKF_Q_VEL * _q_scale
        self._R_dist = (EKF_R_DIST * _q_scale) if exact_mode else EKF_R_DIST
        self._innov_hist = {s: [] for s in anchor_positions}

        self.pos_x: Optional[float] = None
        self.pos_y: Optional[float] = None
        self.pos_z: Optional[float] = None
        self.pos_ok: bool = False

    @staticmethod
    def _joseph_update(P, K, H, R):
        n = P.shape[0]
        IKH = np.eye(n) - K @ H
        return IKH @ P @ IKH.T + K @ R @ K.T

    @staticmethod
    def _make_F(dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return F

    def _make_Q(self, dt: float) -> np.ndarray:
        Q = np.zeros((6, 6))
        q_pos = max(self._q_vel * dt ** 2, self._q_pos)
        Q[0, 0] = Q[1, 1] = Q[2, 2] = q_pos
        Q[3, 3] = Q[4, 4] = Q[5, 5] = self._q_vel
        return Q

    @staticmethod
    def _H_jacobian(x: np.ndarray, anchor_pos: np.ndarray) -> np.ndarray:
        diff = x[0:3] - anchor_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            dist = 1e-9
        H = np.zeros((1, 6))
        H[0, 0:3] = diff / dist
        return H

    @staticmethod
    def _h(x: np.ndarray, anchor_pos: np.ndarray) -> float:
        return float(np.linalg.norm(x[0:3] - anchor_pos))

    def _adaptive_R(self, side: str, innov: float) -> float:
        hist = self._innov_hist.get(side, [])
        hist.append(innov)
        if len(hist) > self._INNOV_WINDOW:
            hist.pop(0)
        self._innov_hist[side] = hist
        if len(hist) >= 5:
            return max(self._R_dist, float(np.var(hist)))
        return self._R_dist

    def _sigma_estimate(self, side: str) -> float:
        """Desviación típica estimada del ruido de distancia de `side`, SIN
        mutar el histórico (a diferencia de _adaptive_R, que sí lo hace al
        registrar una innovación nueva). Se usa para ponderar cada ancla en
        _lm_solve según lo fiable que ha resultado hasta ahora: una ancla
        con más ruido reciente pesa menos en la trilateración."""
        hist = self._innov_hist.get(side, [])
        if len(hist) >= 5:
            return math.sqrt(max(self._R_dist, float(np.var(hist))))
        return math.sqrt(self._R_dist)

    def _lm_solve(self, measurements: list, interior_hint=None) -> Optional[np.ndarray]:
        from statistics import median
        by_side = {}
        for s, d in measurements:
            if s in self.anchor_positions and EKF_MIN_DIST < d < EKF_MAX_DIST:
                by_side.setdefault(s, []).append(d)
        unique = {s: median(ds) for s, ds in by_side.items()}

        if len(unique) < 1:
            return None

        anchors_used = [(self.anchor_positions[s], d) for s, d in unique.items()]

        if len(unique) == 1:
            a_pos, d_m = anchors_used[0]
            bed_center = np.mean(list(self.anchor_positions.values()), axis=0)
            direction = bed_center - a_pos
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction /= norm
            return a_pos + direction * min(d_m, norm)

        if len(unique) >= 3:
            sigmas = [self._sigma_estimate(s) for s in unique.keys()]
            return _solve_trilateration(anchors_used, interior_hint=interior_hint, sigmas=sigmas)
        else:
            z_fixed = float(np.mean([a[2] for a, _ in anchors_used]))

            def residuals2(p2):
                p3 = np.array([p2[0], p2[1], z_fixed])
                return [np.linalg.norm(p3 - a) - d for a, d in anchors_used]

            p0_2 = np.mean([a[:2] for a, _ in anchors_used], axis=0)
            try:
                res = least_squares(residuals2, p0_2, method="lm", max_nfev=400)
                return np.array([res.x[0], res.x[1], z_fixed])
            except Exception:
                return np.array([p0_2[0], p0_2[1], z_fixed])

    def predict(self, dt: float = 0.1) -> None:
        if not self._initialized:
            return
        if dt <= 0 or dt > 0.5:
            return
        F = self._make_F(dt)
        Q = self._make_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def correct(self, side: str, dist_m: float) -> None:
        if side not in self.anchor_positions:
            return
        if not self._initialized:
            self._warmup_buf.append((side, dist_m))
            distinct = len({s for s, _ in self._warmup_buf})
            buf_len = len(self._warmup_buf)
            enough = buf_len >= EKF_WARMUP_SAMPLES and distinct >= 2
            enough_fallback = buf_len >= EKF_WARMUP_SAMPLES * 3 and distinct >= 1
            if enough or enough_fallback:
                p0 = self._lm_solve(self._warmup_buf)
                if p0 is not None:
                    self.x[0:3] = p0
                    self.x[3:6] = 0.0
                    self.P = np.diag([0.01, 0.01, 0.01, 0.5, 0.5, 0.5])
                    self._initialized = True
                    self.pos_x = float(p0[0])
                    self.pos_y = float(p0[1])
                    self.pos_z = float(p0[2])
                    self.pos_ok = True
                else:
                    self._warmup_buf.clear()
            return

        dist_m_clipped = float(np.clip(dist_m, EKF_MIN_DIST, EKF_MAX_DIST))
        anchor_pos = self.anchor_positions[side]

        h_val = self._h(self.x, anchor_pos)
        innov = dist_m_clipped - h_val

        R_val = self._adaptive_R(side, innov)
        R = np.array([[R_val]])
        H = self._H_jacobian(self.x, anchor_pos)
        S = H @ self.P @ H.T + R

        chi2 = float((innov ** 2) / S[0, 0])
        if chi2 > self._CHI2_GATE:
            self.pos_x = float(self.x[0])
            self.pos_y = float(self.x[1])
            self.pos_z = float(self.x[2])
            self.pos_ok = True
            return

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ np.array([[innov]])).flatten()
        self.P = self._joseph_update(self.P, K, H, R)

        self.pos_x = float(self.x[0])
        self.pos_y = float(self.x[1])
        self.pos_z = float(self.x[2])
        self.pos_ok = True

    def correct_batch(self, measurements: list) -> None:
        """Actualización CONJUNTA del EKF con todas las medidas ancla-tag
        del instante actual, en vez de aplicar correct() ancla a ancla en un
        bucle secuencial.

        Por qué esto ayuda quando aumenta el nº de anclas: con correct()
        secuencial, la 2ª ancla se lineliza (Jacobiano H) y se filtra (gate
        chi2) sobre un estado x que YA ha sido desplazado por la corrección
        de la 1ª ancla; la 3ª sobre el estado ya desplazado por la 1ª y la
        2ª; etc. Al ser h(x) no lineal, ese desplazamiento del punto de
        linealización de una corrección a la siguiente introduce un sesgo
        que depende del ORDEN en que llegan las anclas y que crece con el
        nº de anclas fusionadas en el mismo instante — justo el efecto de
        "más anclas, más error" cuando ese efecto domina sobre la ganancia
        de información.

        Aquí, en cambio, TODAS las medidas del instante se linealizan sobre
        el MISMO estado previo (antes de esta actualización), se apilan en
        un único H (m x 6) / R (m x m) y se resuelven con una única
        ganancia de Kalman K = P Hᵀ (H P Hᵀ + R)⁻¹. Esto es la actualización
        de medida conjunta correcta de un EKF con medidas simultáneas: cada
        ancla aporta información en proporción a su geometría (H) y a su
        fiabilidad reciente (R adaptativa), sin que el orden de llegada ni
        el nº de anclas introduzcan sesgo adicional. El filtrado de
        outliers (chi2 > _CHI2_GATE) se conserva igual que en correct(),
        pero también se evalúa sobre el estado previo común, no sobre
        estados intermedios ya corregidos."""
        if not measurements:
            return

        valid = []
        for side, dist_m in measurements:
            if side not in self.anchor_positions:
                continue
            dist_m_clipped = float(np.clip(dist_m, EKF_MIN_DIST, EKF_MAX_DIST))
            anchor_pos = self.anchor_positions[side]

            h_val = self._h(self.x, anchor_pos)
            innov = dist_m_clipped - h_val
            H_i = self._H_jacobian(self.x, anchor_pos)
            R_i = self._adaptive_R(side, innov)
            S_i = float((H_i @ self.P @ H_i.T).item()) + R_i
            chi2_i = (innov ** 2) / S_i

            if chi2_i > self._CHI2_GATE:
                continue  # medida individual descartada por outlier, igual que en correct()
            valid.append((H_i, R_i, innov))

        if not valid:
            # ninguna medida pasó el gate en este instante: se mantiene la
            # predicción, igual que hacía correct() cuando chi2 > gate
            self.pos_x = float(self.x[0])
            self.pos_y = float(self.x[1])
            self.pos_z = float(self.x[2])
            self.pos_ok = True
            return

        H = np.vstack([h for h, _, _ in valid])            # (m, 6)
        R = np.diag([r for _, r, _ in valid])               # (m, m)
        y = np.array([i for _, _, i in valid])              # (m,)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)                  # (6, m)
        self.x = self.x + (K @ y)
        self.P = self._joseph_update(self.P, K, H, R)

        self.pos_x = float(self.x[0])
        self.pos_y = float(self.x[1])
        self.pos_z = float(self.x[2])
        self.pos_ok = True

    def init_from_instant(self, distances: list, interior_hint=None) -> bool:
        p0 = self._lm_solve(distances, interior_hint=interior_hint)
        if p0 is None:
            return False
        self.x[0:3] = p0
        self.x[3:6] = 0.0
        self.P = np.diag([1e-8, 1e-8, 1e-8, 0.5, 0.5, 0.5])
        self._initialized = True
        self.pos_x = float(p0[0])
        self.pos_y = float(p0[1])
        self.pos_z = float(p0[2])
        self.pos_ok = True
        return True


def estimate_positions_lm_only(anchor_positions: dict, trajectory_points: list,
                                noisy_distances: list, interior_hint=None) -> list:
    results = []
    for pt_idx, point in enumerate(trajectory_points):
        distances = noisy_distances[pt_idx]
        anchors_used = []
        for name, dist in distances:
            if name in anchor_positions and dist > EKF_MIN_DIST:
                anchors_used.append((anchor_positions[name], dist))

        if len(anchors_used) == 0:
            results.append(None)
            continue
        if len(anchors_used) == 1:
            a_pos, d = anchors_used[0]
            center = np.mean(list(anchor_positions.values()), axis=0)
            direction = center - a_pos
            norm = np.linalg.norm(direction)
            direction = np.array([1.0, 0.0, 0.0]) if norm < 1e-9 else direction / norm
            results.append(a_pos + direction * d)
            continue
        if len(anchors_used) >= 3:
            results.append(_solve_trilateration(anchors_used, interior_hint=interior_hint))
        else:
            z_fixed = float(np.mean([a[2] for a, _ in anchors_used]))

            def residuals2(p2):
                p3 = np.array([p2[0], p2[1], z_fixed])
                return [np.linalg.norm(p3 - a) - d for a, d in anchors_used]

            p0_2 = np.mean([a[:2] for a, _ in anchors_used], axis=0)
            try:
                res = least_squares(residuals2, p0_2, method="lm", max_nfev=400)
                results.append(np.array([res.x[0], res.x[1], z_fixed]))
            except Exception:
                results.append(np.array([p0_2[0], p0_2[1], z_fixed]))
    return results


def run_ekf_sequential(anchor_positions: dict, trajectory_points: list, noisy_distances: list,
                        dt_between_points: float = 0.1, exact_mode: bool = False,
                        interior_hint=None) -> list:
    ekf = PositionEKF(anchor_positions, exact_mode=exact_mode)
    results = []
    n = len(trajectory_points)

    for pt_idx in range(n):
        if not ekf._initialized:
            if not ekf.init_from_instant(noisy_distances[pt_idx], interior_hint=interior_hint):
                results.append(None)
                continue
        else:
            dt = dt_between_points
            v_est = (trajectory_points[pt_idx] - trajectory_points[pt_idx - 1]) / dt
            ekf.x[3:6] = v_est
            ekf.predict(dt=dt)
            ekf.correct_batch(noisy_distances[pt_idx])

        results.append(np.array([ekf.pos_x, ekf.pos_y, ekf.pos_z]) if ekf.pos_ok else None)

    return results


# ═══════════════════════════════════════════════════════════════════
# GENERACIÓN DE TRAYECTORIAS (idéntico a la versión anterior)
# ═══════════════════════════════════════════════════════════════════

def generate_trajectory(mode: str, room: dict, n_pts: int = 60, **params) -> list:
    """
    Genera una trayectoria paramétrica dentro de la sala. Modos disponibles:
    'line', 'circle', 'helix', 'lemniscate' (mismas fórmulas que el modo
    dibujo/paramétrico del frontend, sin depender del canvas).
    """
    sx, sy, sz = room["x"], room["y"], room["z"]
    pts = []

    if mode == "line":
        x0, y0, z0 = params.get("p0", (sx * 0.1, sy * 0.1, sz * 0.3))
        x1, y1, z1 = params.get("p1", (sx * 0.9, sy * 0.9, sz * 0.7))
        for i in range(n_pts):
            t = i / (n_pts - 1)
            pts.append(np.array([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0 + (z1 - z0) * t]))

    elif mode == "circle":
        cx, cy, cz = params.get("center", (sx / 2, sy / 2, sz / 2))
        r = params.get("r", min(sx, sy) * 0.3)
        axis = params.get("axis", "z")
        for i in range(n_pts):
            t = (i / n_pts) * 2 * math.pi
            if axis == "z":
                pts.append(np.array([cx + r * math.cos(t), cy + r * math.sin(t), cz]))
            elif axis == "x":
                pts.append(np.array([cx, cy + r * math.cos(t), cz + r * math.sin(t)]))
            else:
                pts.append(np.array([cx + r * math.cos(t), cy, cz + r * math.sin(t)]))

    elif mode == "helix":
        cx, cy = params.get("center_xy", (sx / 2, sy / 2))
        r = params.get("r", min(sx, sy) * 0.3)
        turns = params.get("turns", 2)
        z0, z1 = params.get("z_range", (sz * 0.1, sz * 0.9))
        for i in range(n_pts):
            t = i / (n_pts - 1)
            a = t * turns * 2 * math.pi
            pts.append(np.array([cx + r * math.cos(a), cy + r * math.sin(a), z0 + (z1 - z0) * t]))

    elif mode == "lemniscate":
        cx, cy, cz = params.get("center", (sx / 2, sy / 2, sz / 2))
        a = params.get("a", min(sx, sy) * 0.3)
        b = params.get("b", sz * 0.2)
        for i in range(n_pts):
            t = (i / n_pts) * 2 * math.pi
            d = 1 + math.sin(t) * math.sin(t)
            pts.append(np.array([
                cx + a * math.cos(t) / d,
                cy + a * math.sin(t) * math.cos(t) / d,
                cz + b * math.sin(t * 2) * 0.5,
            ]))
    else:
        raise ValueError(f"Modo de trayectoria desconocido: {mode}")

    return pts


# ═══════════════════════════════════════════════════════════════════
# ERROR EN LA MEDIDA DE DISTANCIA ANCLA-TAG
# Ahora cada escenario es una LISTA de "componentes" de error que se
# aplican de forma acumulativa sobre la distancia real, para poder
# combinar varios tipos de error a la vez (p. ej. bias + gaussiano).
# Cada componente indica explícitamente el signo/rango, para poder
# distinguir sesgo positivo de negativo en vez de asumir uno solo.
# ═══════════════════════════════════════════════════════════════════

def comp_bias(value: float) -> dict:
    """Error constante (bias). value puede ser positivo o negativo."""
    return {"type": "constant", "value": float(value)}


def comp_gauss(mean: float, std: float) -> dict:
    """Error gaussiano N(mean, std). mean=0 -> puramente aleatorio y
    simétrico; mean!=0 -> ruido aleatorio con sesgo sistemático añadido."""
    return {"type": "gaussian", "mean": float(mean), "std": float(std)}


def comp_unif(low: float, high: float) -> dict:
    """Error uniforme en [low, high]. (-v/2, v/2) es simétrico; (0, v) o
    (-v, 0) modela p. ej. el bias casi siempre positivo por multipath/NLOS
    típico de medidas UWB reales."""
    return {"type": "uniform", "low": float(low), "high": float(high)}


def apply_noise_multi(value: float, components: list) -> float:
    """Aplica secuencialmente 0, 1 o varios componentes de error sobre una
    medida de distancia real. Lista vacía == sin error ('none')."""
    result = value
    for comp in components:
        ctype = comp.get("type", "none")
        if ctype == "constant":
            result += float(comp.get("value", 0.0))
        elif ctype == "gaussian":
            mean = float(comp.get("mean", 0.0))
            std = float(comp.get("std", 0.0))
            if std <= 0:
                std = abs(mean) * 0.1 + 0.001 if mean != 0 else 0.001
            result += np.random.normal(mean, std)
        elif ctype == "uniform":
            low = float(comp.get("low", -0.01))
            high = float(comp.get("high", 0.01))
            if high < low:
                low, high = high, low
            result += np.random.uniform(low, high)
        # tipo "none" o desconocido -> no aplica nada
    return result


def _fmt_num(v) -> str:
    """Formatea un número para usarlo en un id/nombre de fichero:
    0.02 -> '0p02', -0.05 -> 'm0p05'."""
    s = f"{v:g}"
    return s.replace("-", "m").replace(".", "p")


def build_noise_scenarios() -> list:
    """
    Construye la lista de escenarios de error a probar en la FASE 2.
    Cada escenario: {"id": str, "components": [comp, ...]}.
    Incluye:
      - "sin_error": baseline sin ruido (para comparar en la misma tabla).
      - bias +/- para cada magnitud de BIAS_MAGNITUDES.
      - gaussiano de media 0 para cada sigma de GAUSS_STDS (simétrico por
        construcción: valores por encima y por debajo del real).
      - uniforme simétrico, solo-positivo y solo-negativo para cada rango
        de UNIFORM_RANGES.
      - combinaciones curadas: bias(+/-) + gaussiano, bias(+) + uniforme
        simétrico, gaussiano + uniforme simétrico.
    """
    scenarios = [{"id": "sin_error", "components": []}]

    for v in BIAS_MAGNITUDES:
        scenarios.append({"id": f"bias_pos_{_fmt_num(v)}", "components": [comp_bias(+v)]})
        scenarios.append({"id": f"bias_neg_{_fmt_num(v)}", "components": [comp_bias(-v)]})

    for s in GAUSS_STDS:
        scenarios.append({"id": f"gauss_s{_fmt_num(s)}", "components": [comp_gauss(0.0, s)]})

    for v in UNIFORM_RANGES:
        scenarios.append({"id": f"unif_sym_{_fmt_num(v)}", "components": [comp_unif(-v / 2, v / 2)]})
        scenarios.append({"id": f"unif_pos_{_fmt_num(v)}", "components": [comp_unif(0.0, v)]})
        scenarios.append({"id": f"unif_neg_{_fmt_num(v)}", "components": [comp_unif(-v, 0.0)]})

    for bv in COMBO_BIAS_MAGNITUDES:
        for s in COMBO_GAUSS_STDS:
            scenarios.append({
                "id": f"combo_biasp{_fmt_num(bv)}_gausss{_fmt_num(s)}",
                "components": [comp_bias(+bv), comp_gauss(0.0, s)],
            })
            scenarios.append({
                "id": f"combo_biasn{_fmt_num(bv)}_gausss{_fmt_num(s)}",
                "components": [comp_bias(-bv), comp_gauss(0.0, s)],
            })

    for bv in COMBO_BIAS_MAGNITUDES:
        for v in COMBO_UNIFORM_RANGES:
            scenarios.append({
                "id": f"combo_biasp{_fmt_num(bv)}_unifsym{_fmt_num(v)}",
                "components": [comp_bias(+bv), comp_unif(-v / 2, v / 2)],
            })
            scenarios.append({
                "id": f"combo_biasn{_fmt_num(bv)}_unifsym{_fmt_num(v)}",
                "components": [comp_bias(-bv), comp_unif(-v / 2, v / 2)],
            })

    for s in COMBO_GAUSS_STDS:
        for v in COMBO_UNIFORM_RANGES:
            scenarios.append({
                "id": f"combo_gausss{_fmt_num(s)}_unifsym{_fmt_num(v)}",
                "components": [comp_gauss(0.0, s), comp_unif(-v / 2, v / 2)],
            })

    return scenarios


def scenario_is_stochastic(components) -> bool:
    """True si el escenario incluye algún componente aleatorio (gaussiano o
    uniforme), y por tanto conviene repetir la simulación para promediar.
    Admite tanto el formato "uniforme" (lista de componentes aplicada por
    igual a todas las anclas) como el formato "heterogéneo"
    ({"per_anchor": {...}, "tag": [...]})."""
    if isinstance(components, dict):
        all_comps = list(components.get("tag", []))
        for lst in components.get("per_anchor", {}).values():
            all_comps.extend(lst)
        return any(c.get("type") in ("gaussian", "uniform") for c in all_comps)
    return any(c.get("type") in ("gaussian", "uniform") for c in components)


# ═══════════════════════════════════════════════════════════════════
# ESCENARIOS DE ERROR HETEROGÉNEOS POR ANCLA + ERROR PROPIO DEL TAG
# A diferencia de build_noise_scenarios() (mismo tipo de error para TODAS
# las anclas a la vez), aquí cada ancla puede sufrir un tipo de error
# distinto e independiente de las demás (p.ej. A1 con gaussiano positivo,
# A2 con uniforme negativo, A3 con bias positivo...), y además el TAG
# aporta su propio error (mismo catálogo) que se suma a la medida de
# distancia con TODAS las anclas por igual (representa, p.ej., una
# imprecisión propia de la antena/electrónica del tag o un drift de reloj,
# distinto del error de cada enlace ancla-tag).
# ═══════════════════════════════════════════════════════════════════

ANCHOR_ERROR_CATALOG = {
    "gaussP": lambda: comp_gauss(+HETERO_GAUSS_MEAN, HETERO_GAUSS_STD),
    "gaussN": lambda: comp_gauss(-HETERO_GAUSS_MEAN, HETERO_GAUSS_STD),
    "gauss0": lambda: comp_gauss(0.0, HETERO_GAUSS_STD),
    "unifP": lambda: comp_unif(0.0, HETERO_UNIFORM_RANGE),
    "unifN": lambda: comp_unif(-HETERO_UNIFORM_RANGE, 0.0),
    "unifS": lambda: comp_unif(-HETERO_UNIFORM_RANGE / 2, HETERO_UNIFORM_RANGE / 2),
    "biasP": lambda: comp_bias(+HETERO_BIAS_MAGNITUDE),
    "biasN": lambda: comp_bias(-HETERO_BIAS_MAGNITUDE),
    "none": lambda: None,
}
# El tag usa el mismo catálogo de tipos de error que las anclas.
TAG_ERROR_CATALOG = ANCHOR_ERROR_CATALOG


def build_heterogeneous_scenarios(anchor_names: list, rng: np.random.Generator,
                                   n_scenarios: int) -> list:
    """Genera escenarios donde cada ancla recibe, de forma independiente,
    un tipo de error del catálogo (bias/gaussiano/uniforme, positivo,
    negativo, simétrico o ninguno), y el tag añade su propio error del
    mismo catálogo sobre todas las medidas por igual.

    El nº de combinaciones posibles es len(catálogo)^(nº_anclas + 1): para
    9 tipos de error y 6 anclas serían 9**7 ≈ 4.8 millones de casos,
    inviable de enumerar (cada uno exige una simulación EKF completa sobre
    varias trayectorias, con repeticiones por ser estocástico). Por eso se
    muestrean aleatoriamente `n_scenarios` combinaciones distintas (sin
    repetir combinación) por cada (nº anclas, sala): es la forma de tener
    'tantos casos como sea posible' cubriendo la interacción entre anclas
    con distintos tipos de error a la vez, sin que el barrido sea
    computacionalmente inabordable. Sube HETERO_SCENARIOS_PER_CONFIG si
    quieres más cobertura (a costa de más tiempo de cómputo)."""
    catalog_names = list(ANCHOR_ERROR_CATALOG.keys())
    seen = set()
    scenarios = []
    attempts = 0
    max_attempts = n_scenarios * 20

    while len(scenarios) < n_scenarios and attempts < max_attempts:
        attempts += 1
        anchor_choice = {name: rng.choice(catalog_names) for name in anchor_names}
        tag_choice = rng.choice(catalog_names)
        key = (tuple(anchor_choice[name] for name in anchor_names), tag_choice)
        if key in seen:
            continue
        seen.add(key)

        per_anchor = {}
        for name in anchor_names:
            comp = ANCHOR_ERROR_CATALOG[anchor_choice[name]]()
            per_anchor[name] = [comp] if comp is not None else []
        tag_comp = TAG_ERROR_CATALOG[tag_choice]()
        tag_components = [tag_comp] if tag_comp is not None else []

        id_parts = [f"{name}-{anchor_choice[name]}" for name in anchor_names]
        id_parts.append(f"tag-{tag_choice}")
        scenario_id = "heter_" + "_".join(id_parts)

        scenarios.append({
            "id": scenario_id,
            "components": {"per_anchor": per_anchor, "tag": tag_components},
        })

    return scenarios


# ═══════════════════════════════════════════════════════════════════
# NÚCLEO DE SIMULACIÓN — equivalente a /api/simulate, sin Flask
# ═══════════════════════════════════════════════════════════════════

def _components_for_anchor(name: str, noise_components) -> list:
    """Devuelve la lista de componentes de error a aplicar a la distancia
    ancla-tag de `name`. Admite el formato "uniforme" (lista, misma para
    todas las anclas) y el formato "heterogéneo" (dict con "per_anchor" y
    "tag", donde el error del tag se suma al de esa ancla en concreto)."""
    if isinstance(noise_components, dict):
        per_anchor = noise_components.get("per_anchor", {}).get(name, [])
        tag_components = noise_components.get("tag", [])
        return list(per_anchor) + list(tag_components)
    return noise_components


def _total_n_components(noise_components) -> int:
    if isinstance(noise_components, dict):
        n = sum(len(v) for v in noise_components.get("per_anchor", {}).values())
        n += len(noise_components.get("tag", []))
        return n
    return len(noise_components)


def run_single_simulation(room: dict, anchors: dict, trajectory_points: list,
                           noise_components, use_ekf: bool = True,
                           sim_duration_s: Optional[float] = None) -> dict:
    real_distances = []
    for pt in trajectory_points:
        real_distances.append([(name, float(np.linalg.norm(pt - apos))) for name, apos in anchors.items()])

    noisy_distances, dist_errors_by_point = [], []
    for pt_dists in real_distances:
        noisy, errors = [], []
        for name, real_d in pt_dists:
            noisy_d = apply_noise_multi(real_d, _components_for_anchor(name, noise_components))
            noisy.append((name, noisy_d))
            errors.append((name, noisy_d - real_d))
        noisy_distances.append(noisy)
        dist_errors_by_point.append(errors)

    n_pts = len(trajectory_points)
    duration = sim_duration_s if sim_duration_s else max(1.0, n_pts * 0.1)
    dt_between_points = duration / max(n_pts - 1, 1)
    exact_mode = _total_n_components(noise_components) == 0
    interior_hint = np.array([room["x"] / 2.0, room["y"] / 2.0, room["z"] / 2.0])

    if use_ekf:
        estimated = run_ekf_sequential(anchors, trajectory_points, noisy_distances,
                                        dt_between_points=dt_between_points, exact_mode=exact_mode,
                                        interior_hint=interior_hint)
    else:
        estimated = estimate_positions_lm_only(anchors, trajectory_points, noisy_distances,
                                                interior_hint=interior_hint)

    estimation_errors = [
        float(np.linalg.norm(real - est)) if est is not None else None
        for real, est in zip(trajectory_points, estimated)
    ]
    valid_errors = [e for e in estimation_errors if e is not None]

    return {
        "trajectory_real": trajectory_points,
        "estimated_positions": estimated,
        "estimation_errors": estimation_errors,
        "real_distances": real_distances,
        "noisy_distances": noisy_distances,
        "dist_errors": dist_errors_by_point,
        "anchor_names": list(anchors.keys()),
        "n_anchors": len(anchors),
        "n_points": n_pts,
        "mean_estimation_error": float(np.mean(valid_errors)) if valid_errors else None,
        "rms_estimation_error": float(np.sqrt(np.mean(np.array(valid_errors) ** 2))) if valid_errors else None,
        "max_estimation_error": float(np.max(valid_errors)) if valid_errors else None,
        "min_estimation_error": float(np.min(valid_errors)) if valid_errors else None,
        "n_valid": len(valid_errors),
    }


def write_result_csv(path: str, room: dict, anchors: dict, sim: dict):
    """Mismo formato de secciones que exportaba el botón CSV del frontend
    (compatible con comparador_global.py)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["=== CONFIGURACIÓN DEL ESPACIO ==="])
        writer.writerow(["Dimensión X (m)", "Dimensión Y (m)", "Dimensión Z (m)"])
        writer.writerow([room["x"], room["y"], room["z"]])
        writer.writerow([])
        writer.writerow(["=== ANCLAS ==="])
        writer.writerow(["Nombre", "X (m)", "Y (m)", "Z (m)"])
        for name, pos in anchors.items():
            writer.writerow([name, pos[0], pos[1], pos[2]])
        writer.writerow([])

        anchor_names = sim["anchor_names"]
        header = ["Punto", "Real_X(m)", "Real_Y(m)", "Real_Z(m)"]
        header += [f"Dist_Exacta_{an}(m)" for an in anchor_names]
        header += [f"Error_Dist_{an}(m)" for an in anchor_names]
        header += ["Est_X(m)", "Est_Y(m)", "Est_Z(m)", "Error_Estimacion(m)"]
        writer.writerow(["=== DATOS POR PUNTO DE TRAYECTORIA ==="])
        writer.writerow(header)

        for i in range(sim["n_points"]):
            real = sim["trajectory_real"][i]
            row = [i + 1, round(real[0], 6), round(real[1], 6), round(real[2], 6)]
            real_d = dict(sim["real_distances"][i])
            err_d = dict(sim["dist_errors"][i])
            row += [round(real_d.get(an, 0.0), 6) for an in anchor_names]
            row += [round(err_d.get(an, 0.0), 6) for an in anchor_names]
            est = sim["estimated_positions"][i]
            if est is not None:
                row += [round(est[0], 6), round(est[1], 6), round(est[2], 6)]
            else:
                row += ["N/A", "N/A", "N/A"]
            ee = sim["estimation_errors"][i]
            row.append(f"{ee:.6e}" if ee is not None else "N/A")
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(["=== RESUMEN ==="])
        writer.writerow(["Nº anclas", sim["n_anchors"]])
        writer.writerow(["Nº puntos trayectoria", sim["n_points"]])
        me = sim["mean_estimation_error"]
        writer.writerow(["Error medio estimación (m)", f"{me:.6e}" if me is not None else "N/A"])


# ═══════════════════════════════════════════════════════════════════
# FASE 1 — BÚSQUEDA DE LA MEJOR COLOCACIÓN DE ANCLAS (sin error)
# ═══════════════════════════════════════════════════════════════════

def generate_anchor_layout(n_anchors: int, room: dict) -> dict:
    """Preset determinista (4 esquinas techo -> puntos medios pared/techo ->
    centro suelo -> interiores techo), ciclando si se piden más de 12. Se usa
    como uno de los candidatos "semilla" de la búsqueda."""
    sx, sy, sz = room["x"], room["y"], room["z"]
    presets = [
        (0, 0, sz), (sx, 0, sz), (0, sy, sz), (sx, sy, sz),
        (sx / 2, 0, sz), (sx / 2, sy, sz), (0, sy / 2, sz), (sx, sy / 2, sz),
        (sx / 2, sy / 2, 0), (sx / 4, sy / 4, sz), (3 * sx / 4, sy / 4, sz), (sx / 4, 3 * sy / 4, sz),
    ]
    anchors = {}
    for i in range(n_anchors):
        x, y, z = presets[i % len(presets)]
        anchors[f"A{i + 1}"] = np.array([x, y, z])
    return anchors


# Layouts adicionales definidos a mano, usados también como semillas de la
# búsqueda cuando son compatibles con la sala para la que se diseñaron.
CUSTOM_ANCHOR_LAYOUTS = {
    "custom_v1": {
        3: {"A1": (0, 0.5, 3), "A2": (5, 0.5, 4), "A3": (2.5, 8, 4)},
        4: {"A1": (2.5, 0, 0.5), "A2": (5, 4, 3), "A3": (2.5, 8, 0.5), "A4": (0, 4, 3)},
        5: {"A1": (2.5, 0, 0.5), "A2": (5, 4, 3), "A3": (2.5, 8, 0.5), "A4": (0, 4, 3), "A5": (2.5, 4, 3)},
        6: {"A1": (2.5, 8, 0.5), "A2": (0, 0, 0.5), "A3": (0, 8, 3), "A4": (5, 0, 0.5), "A5": (2.5, 0, 3), "A6": (5, 8, 3)},
    },
}
LAYOUT_ROOM_COMPAT = {"custom_v1": ["hab_custom"]}


def _random_boundary_point(room: dict, rng: np.random.Generator) -> np.ndarray:
    """Punto aleatorio sobre una de las 6 caras de la sala (pared/techo/
    suelo), que es donde físicamente se instalan las anclas y da mejor
    geometría (GDOP) que un punto interior cualquiera."""
    sx, sy, sz = room["x"], room["y"], room["z"]
    face = rng.integers(0, 6)
    if face == 0:
        return np.array([0.0, rng.uniform(0, sy), rng.uniform(0, sz)])
    if face == 1:
        return np.array([sx, rng.uniform(0, sy), rng.uniform(0, sz)])
    if face == 2:
        return np.array([rng.uniform(0, sx), 0.0, rng.uniform(0, sz)])
    if face == 3:
        return np.array([rng.uniform(0, sx), sy, rng.uniform(0, sz)])
    if face == 4:
        return np.array([rng.uniform(0, sx), rng.uniform(0, sy), 0.0])
    return np.array([rng.uniform(0, sx), rng.uniform(0, sy), sz])


def random_anchor_layout(n_anchors: int, room: dict, rng: np.random.Generator) -> dict:
    return {f"A{i + 1}": _random_boundary_point(room, rng) for i in range(n_anchors)}


def score_layout(anchors: dict, room: dict, trajectories: list, use_ekf: bool) -> float:
    """Peor error de estimación (máximo sobre todos los puntos y todas las
    trayectorias de prueba) que da esta configuración de anclas SIN
    introducir ningún error de distancia. Cuanto más bajo, mejor."""
    worst = 0.0
    for traj_pts in trajectories:
        sim = run_single_simulation(room, anchors, traj_pts, [], use_ekf=use_ekf)
        if sim["n_valid"] < sim["n_points"]:
            return math.inf
        worst = max(worst, sim["max_estimation_error"])
    return worst


def build_search_noise_scenarios(anchor_names: list, rng: np.random.Generator) -> list:
    """Escenarios de error usados DURANTE la búsqueda de anclas (Fase 1)
    para valorar la ROBUSTEZ frente a ruido de cada candidato: unos pocos
    escenarios "uniformes" representativos (bias, gaussiano, uniforme) más
    unas cuantas combinaciones HETEROGÉNEAS (cada ancla con un tipo de
    error distinto + error propio del tag). No es el barrido completo de
    la Fase 2 (sería demasiado caro repetirlo para cada candidato durante
    la búsqueda); es un subconjunto pequeño pero variado que sirve para
    comparar la robustez relativa de distintas geometrías de anclas."""
    scenarios = [
        {"id": "search_bias", "components": [comp_bias(SEARCH_NOISE_BIAS)]},
        {"id": "search_gauss", "components": [comp_gauss(0.0, SEARCH_NOISE_GAUSS_STD)]},
        {"id": "search_unif", "components": [comp_unif(-SEARCH_NOISE_UNIFORM_RANGE / 2, SEARCH_NOISE_UNIFORM_RANGE / 2)]},
    ]
    scenarios += build_heterogeneous_scenarios(anchor_names, rng, SEARCH_NOISE_HETERO_SAMPLES)
    return scenarios


def score_layout_noise(anchors: dict, room: dict, trajectories: list, use_ekf: bool,
                        noise_scenarios: list, seeds: list) -> tuple:
    """Error medio (RMS) de esta configuración de anclas promediado sobre
    varios escenarios de error representativos (ver
    build_search_noise_scenarios), varias trayectorias de prueba y, ahora,
    varias SEMILLAS de ruido independientes (`seeds`).

    Antes se evaluaba con una única semilla fija: cada candidato veía
    exactamente UNA realización de ruido por escenario/trayectoria, así que
    el "error con ruido" de cada candidato tenía la varianza propia de esa
    única muestra aleatoria. Eso es un problema especialmente al comparar
    nº de anclas distintos: con más anclas hay más números aleatorios
    consumidos por punto (uno por ancla), así que aunque la semilla sea la
    misma, la secuencia de ruido que "le toca" a cada ancla ya no es
    comparable 1:1 entre una configuración de 4 y una de 6 anclas. Con una
    sola muestra, bastaba con que a una configuración de 5-6 anclas le
    tocase, por puro azar, una realización de ruido algo peor para que
    pareciese "peor que con menos anclas" sin que la geometría fuese
    realmente peor.

    Ahora se repite la evaluación con varias semillas fijas (`seeds`, las
    MISMAS para todos los candidatos de esta búsqueda, así que la
    comparación entre candidatos sigue siendo determinista y reproducible)
    y se promedia el RMS resultante: es una estimación del error esperado
    bajo ruido con mucha menos varianza de muestreo que una sola
    realización, así que la comparación "más anclas, ¿mejora o no?" se
    apoya en una medida más fiable.

    Devuelve (rms_medio, rms_std) — la desviación típica entre semillas se
    guarda solo a título informativo (para ver en el log cuánto "ruido de
    medición" tenía la métrica), no se usa para decidir."""
    errors = []
    for scenario in noise_scenarios:
        for traj_pts in trajectories:
            for seed in seeds:
                np.random.seed(seed)
                sim = run_single_simulation(room, anchors, traj_pts, scenario["components"], use_ekf=use_ekf)
                if sim["n_valid"] < sim["n_points"] or sim["rms_estimation_error"] is None:
                    return math.inf, math.nan
                errors.append(sim["rms_estimation_error"])
    if not errors:
        return math.inf, math.nan
    return float(np.mean(errors)), float(np.std(errors))


def search_best_layout(n_anchors: int, room_name: str, room: dict, rng: np.random.Generator,
                        use_ekf: bool, log_rows: list, noise_log_rows: list,
                        previous_best_error: float = math.inf,
                        previous_best_noise_error: float = math.inf) -> tuple:
    """Busca la mejor configuración de anclas combinando DOS criterios:

      FASE 1a — sin ruido: se prueban candidatos (semillas conocidas +
        aleatorios sobre los límites de la sala) hasta reunir un POOL de
        hasta SEARCH_NOISE_POOL_SIZE candidatos cuyo error sin ruido sea
        del orden de `effective_target` (el más exigente entre
        SEARCH_TARGET_ERROR y `previous_best_error`, el error ya logrado
        con menos anclas en la misma sala), o hasta agotar
        SEARCH_MAX_TRIALS intentos.

      FASE 1b — con ruido: de entre TODOS los candidatos del pool (todas
        las "posibilidades" que ya cumplen el objetivo sin ruido), se
        evalúa cada uno bajo varios escenarios de error representativos
        (build_search_noise_scenarios: bias, gaussiano, uniforme y
        combinaciones heterogéneas por ancla + tag) y se elige el que dé
        MENOR error medio con ruido. Así, el criterio final de selección
        ya no es solo "el primero que sea casi exacto sin ruido", sino
        "el mejor, sin ruido Y con ruido, de entre los que cumplen el
        objetivo sin ruido" — y se compara (`previous_best_noise_error`)
        con lo logrado con menos anclas para valorar si añadir anclas
        también reduce el error cuando SÍ hay error en las medidas de
        distancia.

    Devuelve (mejor_layout, mejor_error_sin_ruido, mejor_error_con_ruido,
    nº_intentos)."""
    trial_trajectories = [
        generate_trajectory(t, room, n_pts=N_TRAJ_POINTS_SEARCH) for t in SEARCH_TRAJECTORIES
    ]

    # Objetivo efectivo de parada (sin ruido): el más exigente entre el
    # umbral absoluto y el error ya logrado con menos anclas.
    effective_target = min(SEARCH_TARGET_ERROR, previous_best_error)

    anchor_names = [f"A{i + 1}" for i in range(n_anchors)]
    noise_scenarios = build_search_noise_scenarios(anchor_names, rng)

    candidates = [("preset", generate_anchor_layout(n_anchors, room))]
    compat_rooms = LAYOUT_ROOM_COMPAT.get("custom_v1")
    if (not compat_rooms or room_name in compat_rooms) and n_anchors in CUSTOM_ANCHOR_LAYOUTS["custom_v1"]:
        raw = CUSTOM_ANCHOR_LAYOUTS["custom_v1"][n_anchors]
        candidates.append(("custom_v1", {k: np.array(v, dtype=float) for k, v in raw.items()}))

    global_best_anchors, global_best_score = None, math.inf  # fallback si ningún candidato alcanza el objetivo
    pool = []  # candidatos que SÍ alcanzan el objetivo sin ruido -> se valoran bajo ruido
    trial_idx = 0

    def _try(origin: str, cand: dict):
        nonlocal global_best_anchors, global_best_score, trial_idx
        trial_idx += 1
        score = score_layout(cand, room, trial_trajectories, use_ekf)
        log_rows.append({
            "n_anchors": n_anchors, "room": room_name, "trial": trial_idx,
            "origin": origin, "max_error_m": score,
            "previous_best_error_m": previous_best_error,
        })
        if score < global_best_score:
            global_best_score, global_best_anchors = score, cand
        if score <= effective_target:
            pool.append((origin, cand, score))

    for origin, cand in candidates:
        _try(origin, cand)

    while len(pool) < SEARCH_NOISE_POOL_SIZE and trial_idx < SEARCH_MAX_TRIALS:
        _try("aleatorio", random_anchor_layout(n_anchors, room, rng))

    if not pool:
        # Ningún candidato alcanzó el objetivo de precisión sin ruido en
        # SEARCH_MAX_TRIALS intentos: no hay "posibilidades" válidas que
        # evaluar bajo ruido, así que se usa el mejor (sin ruido)
        # encontrado como fallback.
        return global_best_anchors, global_best_score, math.inf, trial_idx

    # FASE 1b: evaluar TODOS los candidatos del pool bajo ruido y quedarse
    # con el de menor error medio con ruido. Cada candidato se evalúa con
    # las MISMAS SEARCH_NOISE_EVAL_SEEDS (varias semillas, no una sola),
    # promediando el RMS entre ellas para reducir la varianza de muestreo
    # de la métrica (ver docstring de score_layout_noise).
    best_anchors, best_nf_score, best_noise_score = None, math.inf, math.inf
    for origin, cand, nf_score in pool:
        noise_score, noise_std = score_layout_noise(cand, room, trial_trajectories, use_ekf,
                                                      noise_scenarios, seeds=SEARCH_NOISE_EVAL_SEEDS)
        noise_log_rows.append({
            "n_anchors": n_anchors, "room": room_name, "origin": origin,
            "max_error_sin_ruido_m": nf_score, "rms_error_con_ruido_m": noise_score,
            "rms_error_con_ruido_std_m": noise_std,
            "n_semillas_evaluadas": len(SEARCH_NOISE_EVAL_SEEDS),
            "previous_best_noise_error_m": previous_best_noise_error,
        })
        if noise_score < best_noise_score:
            best_noise_score, best_anchors, best_nf_score = noise_score, cand, nf_score

    return best_anchors, best_nf_score, best_noise_score, trial_idx


def run_layout_search(out_dir: str) -> dict:
    """FASE 1 completa: para cada (n_anclas, sala) busca la mejor
    configuración de anclas combinando el error sin ruido y la robustez
    frente a ruido (ver search_best_layout). Devuelve
    {(n_anclas, sala): anchors} y escribe:
      - busqueda_anclas_log.csv: cada intento de la fase "sin ruido".
      - busqueda_anclas_log_ruido.csv: cada candidato del pool evaluado
        bajo ruido (fase "con ruido").
      - mejores_configuraciones.csv: una fila por (n_anclas, sala).

    Las cuentas de anclas (ANCHOR_COUNTS) se recorren en orden ascendente
    dentro de cada sala, llevando tanto el mejor error sin ruido
    (`previous_best_error`) como el mejor error con ruido
    (`previous_best_noise_error`) logrados con menos anclas, para que la
    búsqueda persiga explícitamente que más anclas reduzca (o al menos no
    empeore) el error de estimación tanto sin ruido como CON ruido en las
    medidas de distancia ancla-tag."""
    rng = np.random.default_rng(SEARCH_RANDOM_SEED)
    log_rows = []
    noise_log_rows = []
    best_layouts = {}
    best_rows = []

    print("=== FASE 1: búsqueda de la mejor colocación de anclas (sin ruido y con ruido) ===")
    for room_name, room in ROOMS.items():
        previous_best_error = math.inf
        previous_best_noise_error = math.inf
        for n_anchors in sorted(ANCHOR_COUNTS):
            anchors, score, noise_score, n_trials = search_best_layout(
                n_anchors, room_name, room, rng, USE_EKF, log_rows, noise_log_rows,
                previous_best_error=previous_best_error,
                previous_best_noise_error=previous_best_noise_error,
            )
            best_layouts[(n_anchors, room_name)] = anchors

            objetivo_absoluto = score <= SEARCH_TARGET_ERROR
            no_empeora_sin_ruido = score <= previous_best_error
            no_empeora_con_ruido = noise_score <= previous_best_noise_error
            if objetivo_absoluto:
                status_sr = "OK (objetivo de precisión alcanzado)"
            elif no_empeora_sin_ruido:
                status_sr = "OK (no empeora respecto a menos anclas)"
            else:
                status_sr = "AVISO (peor que con menos anclas)"
            status_cr = "OK (no empeora respecto a menos anclas)" if no_empeora_con_ruido else "AVISO (peor que con menos anclas)"
            print(f"  {n_anchors} anclas / {room_name}: "
                  f"sin_ruido={score:.3e} m (anterior={previous_best_error:.3e} m) -> {status_sr} | "
                  f"con_ruido={noise_score:.3e} m (anterior={previous_best_noise_error:.3e} m) -> {status_cr} "
                  f"[{n_trials} intentos]")

            row = {
                "n_anchors": n_anchors, "room": room_name,
                "room_x": room["x"], "room_y": room["y"], "room_z": room["z"],
                "n_trials": n_trials,
                "max_error_sin_ruido_m": score,
                "previous_best_error_sin_ruido_m": previous_best_error,
                "objetivo_precision_alcanzado": objetivo_absoluto,
                "no_empeora_sin_ruido_respecto_menos_anclas": no_empeora_sin_ruido,
                "rms_error_con_ruido_m": noise_score,
                "previous_best_error_con_ruido_m": previous_best_noise_error,
                "no_empeora_con_ruido_respecto_menos_anclas": no_empeora_con_ruido,
            }
            for name, pos in anchors.items():
                row[f"{name}_x"] = pos[0]
                row[f"{name}_y"] = pos[1]
                row[f"{name}_z"] = pos[2]
            best_rows.append(row)

            previous_best_error = min(previous_best_error, score)
            previous_best_noise_error = min(previous_best_noise_error, noise_score)

    log_path = os.path.join(out_dir, "busqueda_anclas_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    noise_log_path = os.path.join(out_dir, "busqueda_anclas_log_ruido.csv")
    if noise_log_rows:
        with open(noise_log_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(noise_log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(noise_log_rows)

    best_path = os.path.join(out_dir, "mejores_configuraciones.csv")
    all_keys = []
    for row in best_rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)
    with open(best_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(best_rows)

    print(f"  -> log de búsqueda (sin ruido): {log_path}")
    if noise_log_rows:
        print(f"  -> log de búsqueda (con ruido): {noise_log_path}")
    print(f"  -> mejores configuraciones: {best_path}")
    return best_layouts


# ═══════════════════════════════════════════════════════════════════
# FASE 2 — BARRIDO DE ERRORES SOBRE LAS MEJORES CONFIGURACIONES
# ═══════════════════════════════════════════════════════════════════

def run_noise_sweep(out_dir: str, best_layouts: dict) -> str:
    noise_scenarios = build_noise_scenarios()
    hetero_rng = np.random.default_rng(HETERO_RANDOM_SEED)
    summary_rows = []
    n_run = 0

    print("=== FASE 2: barrido de errores sobre las mejores configuraciones ===")
    for (n_anchors, room_name), anchors in best_layouts.items():
        room = ROOMS[room_name]
        anchor_names = list(anchors.keys())
        hetero_scenarios = build_heterogeneous_scenarios(anchor_names, hetero_rng, HETERO_SCENARIOS_PER_CONFIG)
        all_scenarios = noise_scenarios + hetero_scenarios

        for traj_mode in TRAJECTORIES:
            trajectory = generate_trajectory(traj_mode, room, n_pts=N_TRAJ_POINTS)

            for scenario in all_scenarios:
                is_stochastic = scenario_is_stochastic(scenario["components"])
                n_reps = REPEATS if is_stochastic else 1

                for rep in range(n_reps):
                    sim = run_single_simulation(room, anchors, trajectory, scenario["components"], use_ekf=USE_EKF)

                    rep_tag = f"_rep{rep + 1}" if is_stochastic else ""
                    fname = f"simulacion_uwb_{n_anchors}anclas_{room_name}_{traj_mode}_{scenario['id']}{rep_tag}.csv"
                    write_result_csv(os.path.join(out_dir, fname), room, anchors, sim)

                    summary_rows.append({
                        "n_anchors": n_anchors,
                        "room": room_name,
                        "room_x": room["x"], "room_y": room["y"], "room_z": room["z"],
                        "trajectory": traj_mode,
                        "escenario_error": scenario["id"],
                        "heterogeneo": isinstance(scenario["components"], dict),
                        "n_componentes_error": _total_n_components(scenario["components"]),
                        "repeat": rep + 1,
                        "n_points": sim["n_points"],
                        "n_valid": sim["n_valid"],
                        "mean_error_m": sim["mean_estimation_error"],
                        "rms_error_m": sim["rms_estimation_error"],
                        "max_error_m": sim["max_estimation_error"],
                        "min_error_m": sim["min_estimation_error"],
                        "csv_file": fname,
                    })
                    n_run += 1

    summary_path = os.path.join(out_dir, "resumen_global.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"  -> {n_run} simulaciones, resumen en: {summary_path}")
    return summary_path


# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL BARRIDO — editar aquí los rangos a explorar
# ═══════════════════════════════════════════════════════════════════

ANCHOR_COUNTS = [3, 4, 5, 6]

ROOMS = {
    "hab_custom": {"x": 5, "y": 8, "z": 3},
}

TRAJECTORIES = ["line", "circle", "helix", "lemniscate"]
N_TRAJ_POINTS = 60          # puntos por trayectoria en la FASE 2 (resultados finales)

# --- Parámetros de la búsqueda de anclas (FASE 1) ---
SEARCH_TARGET_ERROR = 1e-12     # objetivo de error (margen de seguridad sobre el ~1e-15 de doble precisión)
SEARCH_MAX_TRIALS = 10000         # nº máximo de configuraciones aleatorias a probar por (n_anclas, sala)
SEARCH_TRAJECTORIES = ["line", "circle"]   # trayectorias usadas para puntuar cada candidato (más rápido)
N_TRAJ_POINTS_SEARCH = 20       # puntos por trayectoria durante la búsqueda (más rápido que N_TRAJ_POINTS)
SEARCH_RANDOM_SEED = 42

# --- Parámetros de la valoración de ROBUSTEZ FRENTE A RUIDO (FASE 1b) ---
# De entre los candidatos que alcanzan el objetivo sin ruido (pool), se
# evalúa cada uno bajo estos escenarios de error y se elige el de menor
# error medio con ruido (ver search_best_layout / score_layout_noise).
SEARCH_NOISE_POOL_SIZE = 30     # nº de candidatos (sin ruido) a evaluar bajo ruido por cada (n_anclas, sala)
SEARCH_NOISE_EVAL_SEEDS = [7, 17, 27, 37, 47]  # semillas fijas (las mismas para
# todos los candidatos): antes se evaluaba con una única semilla, lo que
# hacía que la comparación "con ruido" entre configuraciones (y sobre todo
# entre nº de anclas distinto) dependiese de una sola realización de ruido.
# Ahora score_layout_noise promedia el RMS sobre TODAS estas semillas, dando
# una estimación del error esperado con mucha menos varianza de muestreo.
# Sube esta lista (más semillas) para una estimación aún más estable, a
# costa de más tiempo de cómputo (el coste de la fase 1b escala linealmente
# con el nº de semillas).
SEARCH_NOISE_BIAS = 0.10        # bias (m) del escenario "search_bias"
SEARCH_NOISE_GAUSS_STD = 0.10   # sigma (m) del escenario "search_gauss"
SEARCH_NOISE_UNIFORM_RANGE = 0.20  # anchura (m) del escenario "search_unif"
SEARCH_NOISE_HETERO_SAMPLES = 10  # nº de combinaciones heterogéneas por ancla+tag usadas en la búsqueda

# --- Parámetros del barrido de error (FASE 2) ---
BIAS_MAGNITUDES = [0.02, 0.05, 0.10]        # bias (m), probado en + y en -
GAUSS_STDS = [0.02, 0.05, 0.10]             # sigma (m) del ruido gaussiano, media 0
UNIFORM_RANGES = [0.05, 0.10, 0.20]         # anchura total (m) del ruido uniforme; se prueba simétrico/+/-

# Subconjuntos usados para las combinaciones de dos tipos de error a la vez
# (para no disparar el nº de escenarios, se usan menos valores que arriba)
COMBO_BIAS_MAGNITUDES = [0.02, 0.05]
COMBO_GAUSS_STDS = [0.02, 0.05]
COMBO_UNIFORM_RANGES = [0.05, 0.10]

# --- Parámetros de los escenarios de error HETEROGÉNEOS por ancla + tag ---
# (ver build_heterogeneous_scenarios más arriba). Cada ancla puede tener un
# tipo de error distinto e independiente, y el tag suma su propio error a
# todas las medidas por igual.
HETERO_BIAS_MAGNITUDE = 0.05     # magnitud (m) del bias +/- en el catálogo heterogéneo
HETERO_GAUSS_MEAN = 0.03         # desplazamiento de media (m) para gaussP/gaussN
HETERO_GAUSS_STD = 0.05          # sigma (m) del ruido gaussiano
HETERO_UNIFORM_RANGE = 0.10      # anchura total (m) del ruido uniforme
HETERO_SCENARIOS_PER_CONFIG = 60 # nº de combinaciones aleatorias por (n_anclas, sala); sube para más cobertura
HETERO_RANDOM_SEED = 123         # semilla para que el muestreo sea reproducible

USE_EKF = True           # False para comparar contra LM punto a punto
REPEATS = 3               # repeticiones por combinación con ruido estocástico (gaussian/uniform)


def run_batch(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    best_layouts = run_layout_search(out_dir)
    summary_path = run_noise_sweep(out_dir, best_layouts)

    elapsed = time.time() - t_start
    print(f"Tiempo total: {elapsed:.1f} s")
    print(f"Todo el resultado (CSVs, log de búsqueda, resumen_global.csv) en: {out_dir}")
    return summary_path


if __name__ == "__main__":
    OUT_DIR = "batch_resultados"
    run_batch(OUT_DIR)
