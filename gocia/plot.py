"""
gocia/plot.py

Progress dashboard for a GOCIA genetic algorithm run.

All figures are built with the Plotly Python library — no JavaScript
authoring required.  Each panel is a plain function that takes a pandas
DataFrame and returns a plotly.graph_objects.Figure, so you can import
and call them individually from a notebook or script.

Usage from the command line
---------------------------
    gocia plot                          # writes gocia_progress.html in CWD
    gocia plot --output my_run.html
    gocia plot --run-dir /path/to/run

Usage from Python
-----------------
    from gocia.database.db import GociaDB
    from gocia.plot import (
        fig_gce_vs_generation,
        fig_status_breakdown,
        fig_coverage,
        fig_operator_efficiency,
        build_dashboard,
    )

    with GociaDB("gocia.db") as db:
        df = db.to_dataframe()

    # individual panel
    fig = fig_gce_vs_generation(df)
    fig.show()
    fig.write_image("gce.pdf")   # requires kaleido: pip install kaleido

    # full 2×2 dashboard
    build_dashboard(df, "progress.html")

Dependencies
------------
    plotly     — pip install plotly
    pandas     — pip install pandas
    kaleido    — optional, only needed for write_image() (PNG/PDF/SVG export)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

# Dark theme that matches the GOCIA terminal
BG       = "#07090e"
PAPER    = "#0d1018"
GRID     = "#182030"
TEXT     = "#a8bcd4"
MUTED    = "#304050"

# Status colours
STATUS_COLORS = {
    "converged":  "#3cc890",
    "isomer":     "#6060d8",
    "desorbed":   "#d87030",
    "failed":     "#d84040",
    "duplicate":  "#486070",
    "submitted":  "#306070",
    "pending":    "#283848",
}

# Operator colours
OPERATOR_COLORS = {
    "init":           "#486070",
    "splice":         "#40a0e0",
    "merge":          "#50c8a0",
    "mutate_add":     "#d0a030",
    "mutate_remove":  "#d05040",
    "mutate_displace":"#9050e0",
}

# Species colour cycle for coverage plot
SPECIES_COLORS = [
    "#3cc890", "#50a8e8", "#e8a030", "#e05050",
    "#a050e0", "#e060a0", "#40c8c8", "#c0c040",
]

SELECTABLE = {"converged", "isomer"}


# ---------------------------------------------------------------------------
# Shared layout helper
# ---------------------------------------------------------------------------

def _base_layout(**extra):
    """Return a dict of layout kwargs applying the dark theme."""
    layout = dict(
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=GRID,
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=54, r=20, t=36, b=44),
        xaxis=dict(
            gridcolor=GRID, zerolinecolor=GRID,
            linecolor=MUTED, tickcolor=MUTED,
        ),
        yaxis=dict(
            gridcolor=GRID, zerolinecolor=GRID,
            linecolor=MUTED, tickcolor=MUTED,
        ),
    )
    layout.update(extra)
    return layout


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _selectable_df(df):
    """Return only converged + isomer rows that have a GCE value."""
    mask = (
        df["status"].isin(SELECTABLE)
        & df["grand_canonical_energy"].notna()
    )
    return df[mask].copy()


def _unpack_ads(extra):
    """Return adsorbate_counts dict from an extra_data value."""
    if not extra or not isinstance(extra, dict):
        return {}
    return extra.get("adsorbate_counts", {})


def _ads_label(extra):
    counts = _unpack_ads(extra)
    if not counts:
        return "—"
    return " ".join(f"{sym}:{n}" for sym, n in sorted(counts.items()))


def _total_ads(extra):
    return sum(_unpack_ads(extra).values())


# ---------------------------------------------------------------------------
# Panel 1 — GCE vs generation
# ---------------------------------------------------------------------------

def fig_gce_vs_generation(df):
    """
    Best-so-far GCE (step line) and mean ± 1σ band per generation.

    Only converged + isomer structures contribute to the statistics.
    Individual structure GCEs are shown as a faint scatter underneath so
    you can see the spread of each generation's pool.

    Parameters
    ----------
    df : pandas.DataFrame
        Full DataFrame from GociaDB.to_dataframe().

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import numpy as np
    import plotly.graph_objects as go

    sel = _selectable_df(df)
    fig = go.Figure()

    if sel.empty:
        fig.add_annotation(
            text="no converged structures yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=12, family="IBM Plex Mono, monospace"),
        )
        fig.update_layout(**_base_layout(
            title=dict(text="GCE vs Generation", font=dict(size=12, color=TEXT)),
        ))
        return fig

    gens = sorted(df["generation"].unique())

    # Per-generation stats
    best_so_far = float("inf")
    running_best, mean_vals, upper_vals, lower_vals = [], [], [], []

    for g in gens:
        sub = sel[sel["generation"] == g]["grand_canonical_energy"]
        if not sub.empty:
            best_so_far = min(best_so_far, sub.min())
            mean_vals.append(sub.mean())
            std = sub.std() if len(sub) > 1 else 0.0
            upper_vals.append(sub.mean() + std)
            lower_vals.append(sub.mean() - std)
        else:
            mean_vals.append(None)
            upper_vals.append(None)
            lower_vals.append(None)
        running_best.append(best_so_far if best_so_far < float("inf") else None)

    # ── mean ± σ band ──
    # Build the closed polygon: upper forward, lower reversed
    band_x = gens + gens[::-1]
    band_y = upper_vals + lower_vals[::-1]
    # Filter out None entries (gens with no selectable structures)
    valid = [(x, y) for x, y in zip(band_x, band_y) if y is not None]
    if valid:
        bx, by = zip(*valid)
        fig.add_trace(go.Scatter(
            x=bx, y=by,
            fill="toself",
            fillcolor="rgba(80,140,220,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="mean ± 1σ",
        ))

    # ── mean line ──
    fig.add_trace(go.Scatter(
        x=gens, y=mean_vals,
        mode="lines",
        name="mean GCE",
        line=dict(color="#5090d8", width=1.5, dash="dot"),
        connectgaps=True,
        hovertemplate="gen %{x}<br>mean: %{y:.4f} eV<extra></extra>",
    ))

    # ── individual structure scatter (faint, coloured by operator) ──
    for op, grp in sel.groupby("operator"):
        col = OPERATOR_COLORS.get(str(op), "#506070")
        hover = [
            f"id: {str(r.id)[:8]}<br>"
            f"gen: {r.generation}<br>"
            f"GCE: {r.grand_canonical_energy:.4f} eV<br>"
            f"ads: {_ads_label(r.extra_data)}<br>"
            f"op: {r.operator}"
            for _, r in grp.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=grp["generation"],
            y=grp["grand_canonical_energy"],
            mode="markers",
            name=str(op),
            marker=dict(
                size=5, color=col, opacity=0.55,
                line=dict(width=0),
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    # ── best-so-far step line ──
    fig.add_trace(go.Scatter(
        x=gens, y=running_best,
        mode="lines",
        name="best so far",
        line=dict(color="#3cc890", width=2.5, shape="hv"),
        connectgaps=True,
        hovertemplate="gen %{x}<br>best: %{y:.4f} eV<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title=dict(text="GCE vs Generation", font=dict(size=12, color=TEXT)),
        xaxis=dict(
            title="generation", dtick=1,
            gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED,
        ),
        yaxis=dict(
            title="grand canonical energy (eV)",
            gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=GRID, borderwidth=1,
            font=dict(size=9),
            tracegroupgap=2,
        ),
    ))
    return fig


# ---------------------------------------------------------------------------
# Panel 2 — Status breakdown per generation
# ---------------------------------------------------------------------------

def fig_status_breakdown(df):
    """
    Stacked bar chart: structure counts by status per generation.

    Useful for diagnosing run health — rising failure or desorption rates
    indicate problems with sampling bounds, slab stability, or sanity thresholds.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    gens = sorted(df["generation"].unique())

    status_order = [
        "converged", "isomer", "desorbed",
        "failed", "duplicate", "submitted", "pending",
    ]

    for status in status_order:
        sub = df[df["status"] == status]
        counts = [
            int((sub["generation"] == g).sum())
            for g in gens
        ]
        if not any(counts):
            continue
        fig.add_trace(go.Bar(
            x=gens,
            y=counts,
            name=status,
            marker_color=STATUS_COLORS.get(status, "#507090"),
            hovertemplate="gen %{x}<br>" + status + ": %{y}<extra></extra>",
        ))

    fig.update_layout(**_base_layout(
        title=dict(text="Status per Generation", font=dict(size=12, color=TEXT)),
        barmode="stack",
        xaxis=dict(
            title="generation", dtick=1,
            gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED,
        ),
        yaxis=dict(
            title="structure count",
            gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED,
        ),
    ))
    return fig


# ---------------------------------------------------------------------------
# Panel 3 — Mean adsorbate coverage per generation
# ---------------------------------------------------------------------------

def fig_coverage(df):
    """
    Mean adsorbate count per species in the selectable pool, per generation.

    Shows whether the GA is exploring different coverage regimes or getting
    stuck at a fixed stoichiometry.  One line per species.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    sel = _selectable_df(df)
    fig = go.Figure()

    if sel.empty:
        fig.add_annotation(
            text="no coverage data yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=12, family="IBM Plex Mono, monospace"),
        )
        fig.update_layout(**_base_layout(
            title=dict(text="Mean Coverage", font=dict(size=12, color=TEXT)),
        ))
        return fig

    gens = sorted(df["generation"].unique())

    # Gather all species seen across the whole run
    all_species: set[str] = set()
    for extra in sel["extra_data"]:
        all_species.update(_unpack_ads(extra).keys())

    for i, sym in enumerate(sorted(all_species)):
        col = SPECIES_COLORS[i % len(SPECIES_COLORS)]
        means = []
        for g in gens:
            pool = sel[sel["generation"] == g]
            if pool.empty:
                means.append(None)
            else:
                vals = [_unpack_ads(r.extra_data).get(sym, 0) for _, r in pool.iterrows()]
                means.append(sum(vals) / len(vals))

        fig.add_trace(go.Scatter(
            x=gens,
            y=means,
            name=sym,
            mode="lines+markers",
            line=dict(color=col, width=2),
            marker=dict(size=5, color=col),
            connectgaps=True,
            hovertemplate="gen %{x}<br>" + sym + ": %{y:.2f} atoms/struct<extra></extra>",
        ))

    fig.update_layout(**_base_layout(
        title=dict(text="Mean Coverage (selectable pool)", font=dict(size=12, color=TEXT)),
        xaxis=dict(
            title="generation", dtick=1,
            gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED,
        ),
        yaxis=dict(
            title="mean atoms per structure",
            gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED,
        ),
    ))
    return fig


# ---------------------------------------------------------------------------
# Panel 4 — Operator convergence rate
# ---------------------------------------------------------------------------

def fig_operator_efficiency(df):
    """
    Fraction of offspring per operator that reached converged or isomer,
    plotted per generation.

    Helps spot when a particular operator starts failing more than expected —
    e.g. splice producing structures that always desorb, or mutate_add
    consistently producing clashes that fail the sanity check.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    terminal = {"converged", "isomer", "desorbed", "failed", "duplicate"}
    fig = go.Figure()
    gens = sorted(df["generation"].unique())

    # Only structures with a terminal status and a known operator
    term = df[df["status"].isin(terminal) & df["operator"].notna()]

    all_ops = sorted(term["operator"].unique())
    if not all_ops:
        fig.add_annotation(
            text="no operator data yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=12, family="IBM Plex Mono, monospace"),
        )
        fig.update_layout(**_base_layout(
            title=dict(text="Operator Convergence Rate", font=dict(size=12, color=TEXT)),
        ))
        return fig

    for op in all_ops:
        col = OPERATOR_COLORS.get(op, "#607080")
        rates = []
        for g in gens:
            pool = term[(term["generation"] == g) & (term["operator"] == op)]
            if pool.empty:
                rates.append(None)
            else:
                n_conv = pool["status"].isin(SELECTABLE).sum()
                rates.append(100.0 * n_conv / len(pool))

        fig.add_trace(go.Scatter(
            x=gens,
            y=rates,
            name=op,
            mode="lines+markers",
            line=dict(color=col, width=2),
            marker=dict(size=5, color=col),
            connectgaps=False,
            hovertemplate="gen %{x}<br>" + op + ": %{y:.0f}%<extra></extra>",
        ))

    # 50% reference line
    fig.add_hline(
        y=50,
        line=dict(color=MUTED, width=1, dash="dot"),
        annotation_text="50%",
        annotation_font=dict(color=MUTED, size=9),
    )

    fig.update_layout(**_base_layout(
        title=dict(text="Operator Convergence Rate", font=dict(size=12, color=TEXT)),
        xaxis=dict(
            title="generation", dtick=1,
            gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED,
        ),
        yaxis=dict(
            title="converged (%)",
            range=[0, 105],
            gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED,
        ),
    ))
    return fig


# ---------------------------------------------------------------------------
# Combined 2×2 dashboard
# ---------------------------------------------------------------------------

def build_dashboard(
    df,
    output_path: str | Path = "gocia_progress.html",
    title: str = "GOCIA Run Progress",
    run_label: str = "",
) -> Path:
    """
    Assemble all four panels into a 2×2 subplot dashboard and write HTML.

    Parameters
    ----------
    df : pandas.DataFrame
        Full DataFrame from GociaDB.to_dataframe().
    output_path : str or Path
        Destination HTML file.
    title : str
        Page title embedded in the HTML.
    run_label : str
        Short label shown in the header (e.g. the run directory name).

    Returns
    -------
    Path
        The output file that was written.
    """
    from datetime import datetime
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    n_structs = len(df)
    n_gens    = df["generation"].nunique()

    subplot_titles = [
        "GCE vs Generation",
        "Status per Generation",
        "Mean Coverage",
        "Operator Convergence Rate",
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    panels = [
        (fig_gce_vs_generation(df),  1, 1),
        (fig_status_breakdown(df),   1, 2),
        (fig_coverage(df),           2, 1),
        (fig_operator_efficiency(df),2, 2),
    ]

    seen_names: set[str] = set()

    for sub_fig, row, col in panels:
        for trace in sub_fig.data:
            # Avoid duplicate legend entries across panels
            if trace.name in seen_names:
                trace.showlegend = False
            else:
                if trace.name:
                    seen_names.add(trace.name)
            fig.add_trace(trace, row=row, col=col)

        # Copy axis titles from sub-figures
        sub_layout = sub_fig.layout
        row_col_suffix = "" if (row == 1 and col == 1) else str((row - 1) * 2 + col)
        for axis_key in (f"xaxis{row_col_suffix}", f"yaxis{row_col_suffix}"):
            src_key = axis_key if row_col_suffix else axis_key
            if hasattr(sub_layout, src_key.replace("2","").replace("3","").replace("4","")):
                pass  # plotly subplot axis naming is handled below

    # Apply axis labels explicitly (subplot axes are xaxis, xaxis2, xaxis3, xaxis4)
    axis_props = dict(
        gridcolor=GRID, zerolinecolor=GRID, linecolor=MUTED, tickcolor=MUTED,
    )
    labels = {
        "xaxis":  "generation",
        "yaxis":  "grand canonical energy (eV)",
        "xaxis2": "generation",
        "yaxis2": "structure count",
        "xaxis3": "generation",
        "yaxis3": "mean atoms per structure",
        "xaxis4": "generation",
        "yaxis4": "converged (%)",
    }
    for axis, label in labels.items():
        fig.update_layout(**{axis: dict(title_text=label, **axis_props)})

    # barmode must be set on the combined figure for the status panel
    fig.update_layout(barmode="stack")

    header_meta = (
        f"{run_label}  ·  {n_structs} structures  ·  "
        f"{n_gens} generation{'s' if n_gens != 1 else ''}  ·  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    fig.update_layout(
        height=760,
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
        title=dict(
            text=f"<b>GOCIA</b>  ·  {header_meta}",
            font=dict(size=12, color=TEXT, family="IBM Plex Mono, monospace"),
            x=0.01, xanchor="left",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=GRID,
            borderwidth=1,
            font=dict(size=9),
            tracegroupgap=0,
        ),
        margin=dict(l=54, r=24, t=52, b=44),
    )

    # Subtitle colour for subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=TEXT, family="IBM Plex Mono, monospace")

    output_path = Path(output_path)
    fig.write_html(
        str(output_path),
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )
    return output_path
