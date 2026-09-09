#!/usr/bin/env python3
"""
generar_graficos_ieee.py

Genera figuras y tablas, en inglés y con estilo apto para un artículo del
IEEE Sensors Journal (fuente serif, tamaños de letra grandes y legibles a
tamaño de columna, líneas gruesas, símbolos distinguibles en blanco y negro,
fuentes embebidas como TrueType), a partir del `resumen_global.csv` que
produce `simulacion_batch.py`.

No necesita importar simulacion_batch.py: todo lo que hace falta (número de
anclas, sala, trayectoria, tipo/valor/signo del error) se reconstruye
parseando la columna 'escenario_error' del CSV.

Genera, por cada sala presente en el CSV (subcarpeta outputs/<sala>/):

  FIGURAS (.pdf vectorial para maquetación + .png de alta resolución)
    fig1_baseline_vs_anchors        -> error sin ruido vs nº de anclas
    fig2_bias_vs_anchors            -> error vs nº de anclas, para cada
                                        magnitud de bias, signo + y signo -
    fig3_gaussian_vs_anchors        -> error vs nº de anclas, para cada sigma
    fig4_uniform_vs_anchors         -> error vs nº de anclas, para cada rango
                                        uniforme (simétrico/solo+/solo-)
    fig5_error_vs_magnitude_bias    -> error vs magnitud de bias, una curva
                                        por nº de anclas
    fig6_error_vs_magnitude_gauss   -> ídem para sigma gaussiano
    fig7_error_vs_magnitude_unif    -> ídem para rango uniforme simétrico
    fig8_comparativa_tipos_error    -> barras agrupadas: nº de anclas vs
                                        tipo de error, a la magnitud más alta
                                        probada de cada tipo
    fig9_combinaciones_error        -> error individual vs combinado
                                        (bias+gauss, bias+unif, gauss+unif)

  TABLAS (para pegar directamente en el artículo)
    tabla_resumen.csv / tabla_resumen.tex
        -> tabla compacta: nº de anclas x escenarios representativos
           (sin error, bias +/- máximo, gaussiano máximo, uniforme máximo,
           un ejemplo de combinación), en notación científica.
    tabla_completa.csv
        -> un valor (media/RMS/máx sobre trayectorias y repeticiones) por
           cada combinación (nº de anclas, escenario de error) probada,
           para consulta o gráficos adicionales.

Uso:
    pip install pandas matplotlib
    python generar_graficos_ieee.py resumen_global.csv [--room hab_estandar] [--out figuras_ieee]
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
# ESTILO IEEE
# ═══════════════════════════════════════════════════════════════════

# Ancho de columna / página a doble columna de IEEE Sensors Journal (pulgadas)
IEEE_COL_WIDTH = 3.45
IEEE_PAGE_WIDTH = 7.16

FIGSIZE_1COL = (IEEE_COL_WIDTH, 2.65)
FIGSIZE_2COL = (IEEE_PAGE_WIDTH, 3.05)

# Colores + marcadores + estilos de línea combinados para que las series se
# distingan también en blanco y negro (impresión), no solo por color.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "8"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
# paleta ampliada (basada en Okabe-Ito, apta para daltonismo) + colores extra
# para que, incluso con 9-12 series en una misma figura, cada una tenga una
# combinación color+marcador+trazo que no coincida exactamente con otra.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
          "#E69F00", "#56B4E9", "#000000", "#8C510A",
          "#8E44AD", "#2C7C4B", "#B03A2E", "#5D6D7E"]


def set_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "axes.grid": True,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        # fuentes embebidas como TrueType (Type 42), no Type 3 -> exigido
        # habitualmente por las plantillas de IEEE para el PDF final.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes(ax, ylog=True):
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    if ylog:
        ax.set_yscale("log")
        ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save_fig(fig, out_dir: str, name: str):
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PARSEO DE 'escenario_error' (id generado por simulacion_batch.py)
# ═══════════════════════════════════════════════════════════════════

def _parse_num(token: str) -> float:
    """'0p05' -> 0.05 ; 'm0p05' -> -0.05 (formato usado por simulacion_batch.py)."""
    neg = token.startswith("m")
    if neg:
        token = token[1:]
    token = token.replace("p", ".")
    val = float(token)
    return -val if neg else val


_NUM = r"(?P<v>m?[0-9]+p?[0-9]*)"
_NUM1 = r"(?P<v1>m?[0-9]+p?[0-9]*)"
_NUM2 = r"(?P<v2>m?[0-9]+p?[0-9]*)"

_PATTERNS = [
    (re.compile(r"^sin_error$"), "none"),
    (re.compile(rf"^bias_pos_{_NUM}$"), "bias_pos"),
    (re.compile(rf"^bias_neg_{_NUM}$"), "bias_neg"),
    (re.compile(rf"^gauss_s{_NUM}$"), "gauss"),
    (re.compile(rf"^unif_sym_{_NUM}$"), "unif_sym"),
    (re.compile(rf"^unif_pos_{_NUM}$"), "unif_pos"),
    (re.compile(rf"^unif_neg_{_NUM}$"), "unif_neg"),
    (re.compile(rf"^combo_biasp{_NUM1}_gausss{_NUM2}$"), "combo_bias_pos_gauss"),
    (re.compile(rf"^combo_biasn{_NUM1}_gausss{_NUM2}$"), "combo_bias_neg_gauss"),
    (re.compile(rf"^combo_biasp{_NUM1}_unifsym{_NUM2}$"), "combo_bias_pos_unif"),
    (re.compile(rf"^combo_biasn{_NUM1}_unifsym{_NUM2}$"), "combo_bias_neg_unif"),
    (re.compile(rf"^combo_gausss{_NUM1}_unifsym{_NUM2}$"), "combo_gauss_unif"),
]

# Etiquetas legibles en inglés para leyendas/tablas
KIND_LABELS = {
    "none": "No error",
    "bias_pos": "Bias (+)",
    "bias_neg": "Bias (\u2212)",
    "gauss": "Gaussian noise",
    "unif_sym": "Uniform noise (symmetric)",
    "unif_pos": "Uniform noise (positive only)",
    "unif_neg": "Uniform noise (negative only)",
    "combo_bias_pos_gauss": "Bias (+) + Gaussian",
    "combo_bias_neg_gauss": "Bias (\u2212) + Gaussian",
    "combo_bias_pos_unif": "Bias (+) + Uniform",
    "combo_bias_neg_unif": "Bias (\u2212) + Uniform",
    "combo_gauss_unif": "Gaussian + Uniform",
}


@dataclass
class ScenarioInfo:
    kind: str
    magnitude: Optional[float]
    magnitude2: Optional[float]
    raw: str


def parse_scenario(scenario_id: str) -> ScenarioInfo:
    for pattern, kind in _PATTERNS:
        m = pattern.match(scenario_id)
        if m:
            gd = m.groupdict()
            if not gd:
                return ScenarioInfo(kind, None, None, scenario_id)
            if "v" in gd:
                return ScenarioInfo(kind, _parse_num(gd["v"]), None, scenario_id)
            return ScenarioInfo(kind, _parse_num(gd["v1"]), _parse_num(gd["v2"]), scenario_id)
    return ScenarioInfo("unknown", None, None, scenario_id)


# ═══════════════════════════════════════════════════════════════════
# CARGA Y AGREGACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════

def load_summary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    parsed = df["escenario_error"].apply(parse_scenario)
    df["kind"] = [p.kind for p in parsed]
    df["magnitude"] = [p.magnitude for p in parsed]
    df["magnitude2"] = [p.magnitude2 for p in parsed]
    return df


def aggregate(df: pd.DataFrame, room: str) -> pd.DataFrame:
    """Promedia sobre trayectorias y repeticiones: una fila por (n_anchors,
    escenario_error). El std entre trayectorias/repeticiones se guarda como
    barra de error."""
    sub = df[df["room"] == room].copy()
    grouped = sub.groupby(["n_anchors", "escenario_error", "kind", "magnitude", "magnitude2"], dropna=False)
    agg = grouped.agg(
        mean_error_m=("mean_error_m", "mean"),
        mean_error_std=("mean_error_m", "std"),
        rms_error_m=("rms_error_m", "mean"),
        max_error_m=("max_error_m", "mean"),
    ).reset_index()
    agg["mean_error_std"] = agg["mean_error_std"].fillna(0.0)
    return agg


# ═══════════════════════════════════════════════════════════════════
# FIGURAS
# ═══════════════════════════════════════════════════════════════════

def fig_baseline_vs_anchors(agg: pd.DataFrame, out_dir: str):
    sub = agg[agg["kind"] == "none"].sort_values("n_anchors")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_1COL)
    ax.errorbar(sub["n_anchors"], sub["mean_error_m"], yerr=sub["mean_error_std"],
                marker="o", color=COLORS[0], capsize=3)
    ax.set_xlabel("Number of anchors")
    ax.set_ylabel("Position estimation error (m)")
    ax.set_title("Baseline error (no distance noise)", pad=8)
    ax.set_xticks(sorted(sub["n_anchors"].unique()))
    style_axes(ax, ylog=True)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig1_baseline_vs_anchors")


def _plot_family_vs_anchors(agg: pd.DataFrame, kinds: list, out_dir: str, name: str,
                             title: str, figsize=FIGSIZE_1COL):
    """Una línea por (kind, magnitude) -> error vs nº de anclas."""
    sub = agg[agg["kind"].isin(kinds)].copy()
    if sub.empty:
        return
    series_keys = sorted(sub[["kind", "magnitude"]].drop_duplicates().itertuples(index=False),
                          key=lambda r: (kinds.index(r.kind), r.magnitude))

    fig, ax = plt.subplots(figsize=figsize)
    for i, key in enumerate(series_keys):
        s = sub[(sub["kind"] == key.kind) & (sub["magnitude"] == key.magnitude)].sort_values("n_anchors")
        label = f"{KIND_LABELS.get(key.kind, key.kind)}, {key.magnitude:g} m"
        ax.errorbar(s["n_anchors"], s["mean_error_m"], yerr=s["mean_error_std"],
                     marker=MARKERS[i % len(MARKERS)], linestyle=LINESTYLES[i % len(LINESTYLES)],
                     color=COLORS[i % len(COLORS)], capsize=3, label=label)

    ax.set_xlabel("Number of anchors")
    ax.set_ylabel("Mean position error (m)")
    ax.set_title(title, fontsize=10.8, pad=8)
    ax.set_xticks(sorted(sub["n_anchors"].unique()))
    style_axes(ax, ylog=True)
    ax.legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save_fig(fig, out_dir, name)


def fig_bias_vs_anchors(agg: pd.DataFrame, out_dir: str):
    _plot_family_vs_anchors(agg, ["bias_pos", "bias_neg"], out_dir,
                             "fig2_bias_vs_anchors",
                             "Effect of distance bias on positioning error",
                             figsize=FIGSIZE_2COL)


def fig_gaussian_vs_anchors(agg: pd.DataFrame, out_dir: str):
    _plot_family_vs_anchors(agg, ["gauss"], out_dir,
                             "fig3_gaussian_vs_anchors",
                             "Effect of Gaussian distance noise on positioning error")


def fig_uniform_vs_anchors(agg: pd.DataFrame, out_dir: str):
    _plot_family_vs_anchors(agg, ["unif_sym", "unif_pos", "unif_neg"], out_dir,
                             "fig4_uniform_vs_anchors",
                             "Effect of uniform distance noise on positioning error",
                             figsize=FIGSIZE_2COL)


def _plot_error_vs_magnitude(agg: pd.DataFrame, kind: str, out_dir: str, name: str, title: str):
    """Una línea por nº de anclas -> error vs magnitud del error introducido."""
    sub = agg[agg["kind"] == kind].copy()
    if sub.empty:
        return
    anchor_counts = sorted(sub["n_anchors"].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE_1COL)
    for i, n in enumerate(anchor_counts):
        s = sub[sub["n_anchors"] == n].sort_values("magnitude")
        ax.errorbar(s["magnitude"], s["mean_error_m"], yerr=s["mean_error_std"],
                     marker=MARKERS[i % len(MARKERS)], linestyle=LINESTYLES[i % len(LINESTYLES)],
                     color=COLORS[i % len(COLORS)], capsize=3, label=f"{n} anchors")

    ax.set_xlabel("Injected distance-error magnitude (m)")
    ax.set_ylabel("Mean position error (m)")
    ax.set_title(title, fontsize=10.8, pad=8)
    style_axes(ax, ylog=True)
    ax.legend(frameon=False, title="Anchor count", loc="center left", bbox_to_anchor=(1.02, 0.5))
    save_fig(fig, out_dir, name)


def fig_error_vs_magnitude(agg: pd.DataFrame, out_dir: str):
    _plot_error_vs_magnitude(agg, "bias_pos", out_dir, "fig5_error_vs_magnitude_bias",
                              "Error vs. bias magnitude (+)")
    _plot_error_vs_magnitude(agg, "gauss", out_dir, "fig6_error_vs_magnitude_gauss",
                              "Error vs. Gaussian noise $\\sigma$")
    _plot_error_vs_magnitude(agg, "unif_sym", out_dir, "fig7_error_vs_magnitude_unif",
                              "Error vs. uniform noise range")


def fig_comparativa_tipos_error(agg: pd.DataFrame, out_dir: str):
    """Barras agrupadas: nº de anclas (grupos) x tipo de error (barras), a la
    magnitud más alta disponible de cada tipo, para comparar de un vistazo
    qué tipo de error afecta más a la precisión."""
    families = ["bias_pos", "bias_neg", "gauss", "unif_sym"]
    rows = []
    for kind in families:
        sub = agg[agg["kind"] == kind]
        if sub.empty:
            continue
        max_mag = sub["magnitude"].max()
        rows.append(sub[np.isclose(sub["magnitude"], max_mag)])
    if not rows:
        return
    sub = pd.concat(rows, ignore_index=True)

    anchor_counts = sorted(sub["n_anchors"].unique())
    kinds_present = [k for k in families if k in sub["kind"].unique()]
    x = np.arange(len(anchor_counts))
    width = 0.8 / max(len(kinds_present), 1)

    fig, ax = plt.subplots(figsize=FIGSIZE_2COL)
    for i, kind in enumerate(kinds_present):
        s = sub[sub["kind"] == kind].set_index("n_anchors").reindex(anchor_counts)
        mag = s["magnitude"].dropna().iloc[0] if s["magnitude"].notna().any() else float("nan")
        label = f"{KIND_LABELS.get(kind, kind)} ({mag:g} m)"
        ax.bar(x + (i - (len(kinds_present) - 1) / 2) * width, s["mean_error_m"], width=width,
               yerr=s["mean_error_std"], capsize=2, color=COLORS[i % len(COLORS)], label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in anchor_counts])
    ax.set_xlabel("Number of anchors")
    ax.set_ylabel("Mean position error (m)")
    ax.set_title("Comparison of error types at their largest tested magnitude")
    style_axes(ax, ylog=True)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    save_fig(fig, out_dir, "fig8_comparativa_tipos_error")


def fig_combinaciones_error(agg: pd.DataFrame, out_dir: str):
    """Compara, para cada nº de anclas, el error de un tipo de error solo
    frente al de la combinación correspondiente (a la magnitud más alta
    común probada en las combinaciones)."""
    combos = [
        ("combo_bias_pos_gauss", "bias_pos", "gauss", "Bias (+) & Gaussian"),
        ("combo_gauss_unif", "gauss", "unif_sym", "Gaussian & Uniform"),
    ]
    anchor_counts = sorted(agg["n_anchors"].unique())
    fig, axes = plt.subplots(1, len(combos), figsize=FIGSIZE_2COL, sharey=True)
    if len(combos) == 1:
        axes = [axes]

    for ax, (combo_kind, kind_a, kind_b, title) in zip(axes, combos):
        combo_sub = agg[agg["kind"] == combo_kind]
        if combo_sub.empty:
            continue
        mag1 = combo_sub["magnitude"].max()
        mag2 = combo_sub[np.isclose(combo_sub["magnitude"], mag1)]["magnitude2"].max()

        combo_s = combo_sub[np.isclose(combo_sub["magnitude"], mag1) &
                             np.isclose(combo_sub["magnitude2"], mag2)].set_index("n_anchors").reindex(anchor_counts)
        a_s = agg[(agg["kind"] == kind_a) & np.isclose(agg["magnitude"], mag1)].set_index("n_anchors").reindex(anchor_counts)
        b_s = agg[(agg["kind"] == kind_b) & np.isclose(agg["magnitude"], mag2)].set_index("n_anchors").reindex(anchor_counts)

        x = np.arange(len(anchor_counts))
        width = 0.26
        ax.bar(x - width, a_s["mean_error_m"], width=width, color=COLORS[0],
               label=f"{KIND_LABELS[kind_a]} only ({mag1:g} m)")
        ax.bar(x, b_s["mean_error_m"], width=width, color=COLORS[1],
               label=f"{KIND_LABELS[kind_b]} only ({mag2:g} m)")
        ax.bar(x + width, combo_s["mean_error_m"], width=width, color=COLORS[2],
               label="Combined")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in anchor_counts])
        ax.set_xlabel("Number of anchors")
        ax.set_title(title)
        style_axes(ax, ylog=True)
        ax.legend(frameon=False, fontsize=8, loc="best")

    axes[0].set_ylabel("Mean position error (m)")
    fig.suptitle("Individual vs. combined distance-error sources", y=1.03)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig9_combinaciones_error")


# ═══════════════════════════════════════════════════════════════════
# TABLAS
# ═══════════════════════════════════════════════════════════════════

def build_summary_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Tabla compacta y representativa: nº de anclas x escenarios clave, con
    la magnitud más alta disponible de cada tipo de error."""
    anchor_counts = sorted(agg["n_anchors"].unique())
    key_kinds = ["none", "bias_pos", "bias_neg", "gauss", "unif_sym"]

    columns = {}
    col_order = []
    for kind in key_kinds:
        sub = agg[agg["kind"] == kind]
        if sub.empty:
            continue
        if kind == "none":
            s = sub.set_index("n_anchors").reindex(anchor_counts)
            col_name = "No error"
        else:
            max_mag = sub["magnitude"].max()
            s = sub[np.isclose(sub["magnitude"], max_mag)].set_index("n_anchors").reindex(anchor_counts)
            col_name = f"{KIND_LABELS[kind]} ({max_mag:g} m)"
        columns[col_name] = s["mean_error_m"]
        col_order.append(col_name)

    # un ejemplo de combinación (bias(+) + gaussiano) a la magnitud más alta común
    combo_sub = agg[agg["kind"] == "combo_bias_pos_gauss"]
    if not combo_sub.empty:
        mag1 = combo_sub["magnitude"].max()
        s = combo_sub[np.isclose(combo_sub["magnitude"], mag1)]
        mag2 = s["magnitude2"].max()
        s = s[np.isclose(s["magnitude2"], mag2)].set_index("n_anchors").reindex(anchor_counts)
        col_name = f"Bias (+{mag1:g} m) + Gaussian ($\\sigma$={mag2:g} m)"
        columns[col_name] = s["mean_error_m"]
        col_order.append(col_name)

    table = pd.DataFrame(columns)[col_order]
    table.index.name = "Anchors"
    return table


def write_summary_table(table: pd.DataFrame, out_dir: str):
    table.to_csv(os.path.join(out_dir, "tabla_resumen.csv"), float_format="%.3e")

    def fmt(v):
        return "--" if pd.isna(v) else f"{v:.2e}"

    latex_table = table.map(fmt).to_latex(
        escape=False,
        caption="Mean position estimation error (m) as a function of the number of "
                "anchors and the type/magnitude of distance-measurement error.",
        label="tab:error_summary",
        column_format="l" + "c" * table.shape[1],
    )
    with open(os.path.join(out_dir, "tabla_resumen.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table)


def write_full_table(agg: pd.DataFrame, out_dir: str):
    full = agg.copy()
    full["error_label"] = full["kind"].map(KIND_LABELS).fillna(full["kind"])
    full = full[["n_anchors", "escenario_error", "error_label", "kind", "magnitude", "magnitude2",
                 "mean_error_m", "mean_error_std", "rms_error_m", "max_error_m"]]
    full = full.sort_values(["n_anchors", "kind", "magnitude", "magnitude2"])
    full.to_csv(os.path.join(out_dir, "tabla_completa.csv"), index=False, float_format="%.6e")


# ═══════════════════════════════════════════════════════════════════
# ORQUESTACIÓN
# ═══════════════════════════════════════════════════════════════════

def generate_all(csv_path: str, out_root: str, rooms_filter: Optional[list] = None):
    set_ieee_style()
    df = load_summary(csv_path)

    rooms = rooms_filter if rooms_filter else sorted(df["room"].unique())
    os.makedirs(out_root, exist_ok=True)

    for room in rooms:
        if room not in df["room"].unique():
            print(f"  [aviso] la sala '{room}' no está en el CSV, se omite.")
            continue
        out_dir = os.path.join(out_root, room)
        os.makedirs(out_dir, exist_ok=True)

        agg = aggregate(df, room)

        fig_baseline_vs_anchors(agg, out_dir)
        fig_bias_vs_anchors(agg, out_dir)
        fig_gaussian_vs_anchors(agg, out_dir)
        fig_uniform_vs_anchors(agg, out_dir)
        fig_error_vs_magnitude(agg, out_dir)
        fig_comparativa_tipos_error(agg, out_dir)
        fig_combinaciones_error(agg, out_dir)

        table = build_summary_table(agg)
        write_summary_table(table, out_dir)
        write_full_table(agg, out_dir)

        print(f"  -> {room}: figuras y tablas en {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Genera figuras y tablas estilo IEEE a partir de resumen_global.csv")
    parser.add_argument("csv_path", nargs="?", default="batch_resultados/resumen_global.csv",
                         help="Ruta a resumen_global.csv (salida de simulacion_batch.py)")
    parser.add_argument("--room", action="append", dest="rooms",
                         help="Sala a procesar (repetible). Por defecto, todas las salas del CSV.")
    parser.add_argument("--out", default="figuras_ieee", help="Carpeta de salida")
    args = parser.parse_args()

    print(f"Leyendo {args.csv_path} ...")
    generate_all(args.csv_path, args.out, rooms_filter=args.rooms)
    print("Listo.")


if __name__ == "__main__":
    main()
