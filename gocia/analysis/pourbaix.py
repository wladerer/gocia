"""
gocia/analysis/pourbaix.py

Coverage phase diagram and Pourbaix diagram from a GOCIA run database.

Theory
------
For each converged structure the grand canonical energy is:

    G(U, pH, T, P) = E_raw - E_slab - Σ_i [ n_i · μ_i(U, pH, T, P) ]

where μ_i includes the CHE electrochemical correction.  At each point in
(U, pH) space, the structure with the lowest G wins and defines the stable
phase.  The collection of winning phases across the space is the Pourbaix
diagram.

The 1D coverage phase diagram is the same calculation swept over a single
chemical potential axis (Δμ_O or Δμ_CO) at fixed (U, pH).

Key concepts
------------
- The bare slab (n_i = 0 for all i) is always included as the reference
  phase.  Its G = E_slab - E_slab = 0 by definition, so it wins wherever
  no adsorbate configuration is stabilised.
- The phase label at each point is the adsorbate composition string,
  e.g. "O:2 CO:1" or "bare".
- We do NOT compute the full convex hull — we take the per-point minimum
  over all structures in the database.  For a well-sampled GA run this is
  equivalent, but unusual structures far from the hull may appear as
  spurious phases in poorly-sampled regions.  Increase population size and
  run length before trusting fine structure in the phase boundaries.

Public API
----------
    coverage_vs_mu(df, cfg, mu_range, species, n_points) -> pd.DataFrame
    pourbaix_grid(df, cfg, U_range, pH_range, n_U, n_pH) -> dict
    fig_coverage_vs_mu(sweep_df) -> plotly.Figure
    fig_pourbaix(grid_dict) -> plotly.Figure
    build_pourbaix_html(db_path, config_path, output_path, ...) -> Path
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Colour palette (consistent with gocia/plot.py)
# ---------------------------------------------------------------------------

BG    = "#07090e"
PAPER = "#0d1018"
GRID  = "#182030"
TEXT  = "#a8bcd4"
MUTED = "#304050"

# Distinct colours for phase regions — up to 16 phases before cycling
PHASE_COLORS = [
    "#3cc890",   # bare / low coverage → green
    "#4090d8",   # blue family
    "#60b8f0",
    "#80d0f8",
    "#d08030",   # orange family
    "#e0b040",
    "#f0d060",
    "#d05040",   # red family
    "#e07060",
    "#9050c0",   # purple family
    "#b070e0",
    "#d090f0",
    "#40c8b0",   # teal family
    "#60e0c0",
    "#c0c040",   # yellow
    "#e0e060",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _selectable(df):
    """Converged + isomer rows with a raw energy."""
    mask = (
        df["status"].isin(["converged", "isomer"])
        & df["raw_energy"].notna()
    )
    return df[mask].copy()


def _ads_label(extra):
    """Return e.g. 'O:2 CO:1' or 'bare'."""
    if not extra or not isinstance(extra, dict):
        return "bare"
    counts = extra.get("adsorbate_counts", {})
    if not counts:
        return "bare"
    return " ".join(f"{sym}:{n}" for sym, n in sorted(counts.items()))


def _ads_counts(extra):
    if not extra or not isinstance(extra, dict):
        return {}
    return extra.get("adsorbate_counts", {})


# ---------------------------------------------------------------------------
# Core sweep: evaluate G for every structure over a condition grid
# ---------------------------------------------------------------------------

def _eval_grid(df, cfg, U_vals, pH_vals, temperature=None, pressure=None):
    """
    For each (U, pH) point find the structure with the lowest G and return
    its phase label and G value.

    Parameters
    ----------
    df : pd.DataFrame
        Full to_dataframe() output.
    cfg : GociaConfig
        Loaded gocia.yaml — supplies chemical_potentials, slab_energy,
        default temperature/pressure.
    U_vals : array-like of float
        Electrode potentials (V vs RHE).
    pH_vals : array-like of float
        pH values.
    temperature : float or None
        Override cfg.conditions.temperature.
    pressure : float or None
        Override cfg.conditions.pressure.

    Returns
    -------
    dict with keys:
        U_vals   : list[float]
        pH_vals  : list[float]
        phases   : list[list[str]]   — phases[i_U][i_pH]
        G_winner : list[list[float]] — lowest G at each point
        all_phases : list[str]       — sorted unique phase labels
        phase_colors : dict[str, str]
    """
    from gocia.fitness.che import grand_canonical_energy

    T = temperature if temperature is not None else cfg.conditions.temperature
    P = pressure    if pressure    is not None else cfg.conditions.pressure

    chem_pots = {ads.symbol: ads.chemical_potential for ads in cfg.adsorbates}
    slab_e    = cfg.slab.energy

    sel = _selectable(df)

    # Pre-extract arrays for speed
    raw_energies   = sel["raw_energy"].tolist()
    ads_counts_list = [_ads_counts(e) for e in sel["extra_data"].tolist()]
    labels          = [_ads_label(e) for e in sel["extra_data"].tolist()]

    phases   = []
    G_winner = []

    for U in U_vals:
        row_phases = []
        row_G      = []
        for pH in pH_vals:
            best_G     = 0.0          # bare slab reference: G = E_slab - E_slab = 0
            best_label = "bare"

            for raw_e, ads_c, label in zip(raw_energies, ads_counts_list, labels):
                # G relative to bare slab
                try:
                    G_abs = grand_canonical_energy(
                        raw_energy=raw_e - slab_e,
                        adsorbate_counts=ads_c,
                        chemical_potentials=chem_pots,
                        potential=U,
                        pH=pH,
                        temperature=T,
                        pressure=P,
                    )
                except KeyError:
                    continue

                if G_abs < best_G:
                    best_G     = G_abs
                    best_label = label

            row_phases.append(best_label)
            row_G.append(best_G)

        phases.append(row_phases)
        G_winner.append(row_G)

    # Collect unique phases and assign colours
    all_phases = sorted({p for row in phases for p in row})
    # Put "bare" first
    if "bare" in all_phases:
        all_phases.remove("bare")
        all_phases = ["bare"] + all_phases

    phase_colors = {
        phase: PHASE_COLORS[i % len(PHASE_COLORS)]
        for i, phase in enumerate(all_phases)
    }

    return {
        "U_vals":       list(U_vals),
        "pH_vals":      list(pH_vals),
        "phases":       phases,
        "G_winner":     G_winner,
        "all_phases":   all_phases,
        "phase_colors": phase_colors,
    }


# ---------------------------------------------------------------------------
# 1-D sweep: coverage vs chemical potential
# ---------------------------------------------------------------------------

def coverage_vs_mu(
    df,
    cfg,
    species: str,
    mu_range: tuple[float, float] = (-3.0, 0.0),
    n_points: int = 400,
    fixed_U: float = 0.0,
    fixed_pH: float = 0.0,
    temperature: float | None = None,
    pressure: float | None = None,
):
    """
    Sweep Δμ for one species and find the most stable phase at each point.

    The x-axis is Δμ = μ - μ_ref, where μ_ref is the standard chemical
    potential of the species from cfg.  This makes 0 the standard-state
    reference and negative values correspond to reducing conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Full to_dataframe() output.
    cfg : GociaConfig
    species : str
        Symbol of the species to sweep (e.g. "O", "CO").
        Must be in cfg.adsorbates.
    mu_range : (float, float)
        Range of Δμ in eV.  Default (-3, 0) covers most practical cases.
    n_points : int
        Number of points along the sweep axis.
    fixed_U, fixed_pH : float
        Fixed electrochemical conditions while sweeping μ.
    temperature, pressure : float or None
        Override cfg.conditions values.

    Returns
    -------
    pd.DataFrame with columns:
        delta_mu   — Δμ relative to standard-state reference (eV)
        mu_abs     — absolute chemical potential (eV)
        phase      — winning composition label at this μ
        G_winner   — lowest G at this μ (eV, relative to bare slab)
        n_ads      — total adsorbate count of the winning structure
    """
    import numpy as np
    import pandas as pd
    from gocia.fitness.che import grand_canonical_energy

    T = temperature if temperature is not None else cfg.conditions.temperature
    P = pressure    if pressure    is not None else cfg.conditions.pressure

    # Find the standard chemical potential for the sweep species
    mu_refs = {ads.symbol: ads.chemical_potential for ads in cfg.adsorbates}
    if species not in mu_refs:
        raise ValueError(
            f"Species '{species}' not found in adsorbates config. "
            f"Available: {sorted(mu_refs.keys())}"
        )
    mu_ref = mu_refs[species]

    slab_e = cfg.slab.energy
    sel    = _selectable(df)

    raw_energies    = sel["raw_energy"].tolist()
    ads_counts_list = [_ads_counts(e) for e in sel["extra_data"].tolist()]
    labels          = [_ads_label(e)  for e in sel["extra_data"].tolist()]

    delta_mus = np.linspace(mu_range[0], mu_range[1], n_points)
    rows = []

    for dmu in delta_mus:
        mu_abs = mu_ref + dmu
        # Temporarily override the species chemical potential for this sweep
        chem_pots_sweep = dict(mu_refs)
        chem_pots_sweep[species] = mu_abs

        best_G     = 0.0
        best_label = "bare"
        best_n     = 0

        for raw_e, ads_c, label in zip(raw_energies, ads_counts_list, labels):
            try:
                G = grand_canonical_energy(
                    raw_energy=raw_e - slab_e,
                    adsorbate_counts=ads_c,
                    chemical_potentials=chem_pots_sweep,
                    potential=fixed_U,
                    pH=fixed_pH,
                    temperature=T,
                    pressure=P,
                )
            except KeyError:
                continue

            if G < best_G:
                best_G     = G
                best_label = label
                best_n     = sum(ads_c.values()) if ads_c else 0

        rows.append({
            "delta_mu": float(dmu),
            "mu_abs":   float(mu_abs),
            "phase":    best_label,
            "G_winner": best_G,
            "n_ads":    best_n,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2-D sweep: Pourbaix grid
# ---------------------------------------------------------------------------

def pourbaix_grid(
    df,
    cfg,
    U_range: tuple[float, float] = (-1.0, 1.0),
    pH_range: tuple[float, float] = (0.0, 14.0),
    n_U: int = 120,
    n_pH: int = 100,
    temperature: float | None = None,
    pressure: float | None = None,
):
    """
    Compute the Pourbaix diagram on a (U, pH) grid.

    Parameters
    ----------
    df : pd.DataFrame
    cfg : GociaConfig
    U_range : (float, float)
        Electrode potential range in V vs RHE.
    pH_range : (float, float)
        pH range.
    n_U, n_pH : int
        Grid resolution along each axis.
    temperature, pressure : float or None

    Returns
    -------
    dict — suitable for passing directly to fig_pourbaix().
        Keys: U_vals, pH_vals, phases, G_winner, all_phases, phase_colors
    """
    import numpy as np

    U_vals  = list(np.linspace(U_range[0],  U_range[1],  n_U))
    pH_vals = list(np.linspace(pH_range[0], pH_range[1], n_pH))

    return _eval_grid(df, cfg, U_vals, pH_vals, temperature, pressure)


# ---------------------------------------------------------------------------
# Figure: 1-D coverage vs Δμ
# ---------------------------------------------------------------------------

def fig_coverage_vs_mu(sweep_df, species: str = ""):
    """
    Plot the stable phase and G as a function of Δμ.

    Two subplots stacked vertically:
      - Top:    G_winner vs Δμ coloured by phase
      - Bottom: phase label as a filled band (one colour per phase)

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of coverage_vs_mu().
    species : str
        Species label for axis titles (e.g. "O", "CO").

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    phases     = sweep_df["phase"].tolist()
    delta_mus  = sweep_df["delta_mu"].tolist()
    G_vals     = sweep_df["G_winner"].tolist()

    unique_phases = []
    for p in phases:
        if p not in unique_phases:
            unique_phases.append(p)
    if "bare" in unique_phases:
        unique_phases.remove("bare")
        unique_phases = ["bare"] + unique_phases

    phase_colors = {
        p: PHASE_COLORS[i % len(PHASE_COLORS)]
        for i, p in enumerate(unique_phases)
    }

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
        shared_xaxes=True,
    )

    # ── Top panel: G_winner coloured by phase ──
    # Split into contiguous segments so each phase gets its own colour
    segments = []
    cur_phase = phases[0]
    seg_x, seg_y = [delta_mus[0]], [G_vals[0]]

    for i in range(1, len(phases)):
        if phases[i] == cur_phase:
            seg_x.append(delta_mus[i])
            seg_y.append(G_vals[i])
        else:
            # Close the segment and start a new one
            segments.append((cur_phase, seg_x, seg_y))
            cur_phase = phases[i]
            # Overlap by one point so lines connect
            seg_x = [delta_mus[i-1], delta_mus[i]]
            seg_y = [G_vals[i-1], G_vals[i]]
    segments.append((cur_phase, seg_x, seg_y))

    shown_in_legend = set()
    for phase, sx, sy in segments:
        col = phase_colors.get(phase, "#607080")
        show = phase not in shown_in_legend
        shown_in_legend.add(phase)
        fig.add_trace(go.Scatter(
            x=sx, y=sy,
            mode="lines",
            name=phase,
            line=dict(color=col, width=2.5),
            showlegend=show,
            legendgroup=phase,
            hovertemplate="Δμ = %{x:.3f} eV<br>G = %{y:.4f} eV<extra>" + phase + "</extra>",
        ), row=1, col=1)

    # G = 0 reference (bare slab)
    fig.add_hline(
        y=0, row=1, col=1,
        line=dict(color=MUTED, width=1, dash="dot"),
        annotation_text="bare slab",
        annotation_font=dict(color=MUTED, size=9),
    )

    # ── Bottom panel: phase band ──
    # For each unique phase, draw a filled rectangle between transition points
    # Build list of (phase, x_start, x_end) transitions
    transitions = []
    cur = phases[0]
    x_start = delta_mus[0]
    for i in range(1, len(phases)):
        if phases[i] != cur:
            transitions.append((cur, x_start, delta_mus[i]))
            cur     = phases[i]
            x_start = delta_mus[i]
    transitions.append((cur, x_start, delta_mus[-1]))

    for phase, xs, xe in transitions:
        col = phase_colors.get(phase, "#607080")
        fig.add_trace(go.Scatter(
            x=[xs, xe, xe, xs, xs],
            y=[0, 0, 1, 1, 0],
            fill="toself",
            fillcolor=col,
            line=dict(color="rgba(0,0,0,0)"),
            mode="lines",
            showlegend=False,
            legendgroup=phase,
            hovertemplate=phase + "<extra></extra>",
            opacity=0.75,
        ), row=2, col=1)

        # Phase label centred in band
        mid_x = (xs + xe) / 2
        fig.add_annotation(
            x=mid_x, y=0.5,
            text=phase,
            xref="x2", yref="y2",
            showarrow=False,
            font=dict(size=9, color="#fff",
                      family="IBM Plex Mono, monospace"),
        )

    x_label = f"Δμ<sub>{species}</sub> (eV)" if species else "Δμ (eV)"

    axis_style = dict(
        gridcolor=GRID, zerolinecolor=GRID,
        linecolor=MUTED, tickcolor=MUTED,
    )

    fig.update_layout(
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
        title=dict(
            text=f"Coverage Phase Diagram  ·  Δμ<sub>{species}</sub> sweep",
            font=dict(size=12, color=TEXT),
            x=0.01, xanchor="left",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=GRID, borderwidth=1,
            font=dict(size=10),
        ),
        height=480,
        margin=dict(l=60, r=20, t=44, b=44),
        xaxis=dict(title=x_label, **axis_style),
        yaxis=dict(title="ΔG relative to bare slab (eV)", **axis_style),
        xaxis2=dict(title=x_label, **axis_style),
        yaxis2=dict(
            showticklabels=False, showgrid=False,
            zeroline=False, range=[0, 1],
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Figure: 2-D Pourbaix diagram
# ---------------------------------------------------------------------------

def fig_pourbaix(grid: dict):
    """
    Plot the Pourbaix diagram as a filled heatmap.

    Parameters
    ----------
    grid : dict
        Output of pourbaix_grid().

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import numpy as np
    import plotly.graph_objects as go

    U_vals    = grid["U_vals"]
    pH_vals   = grid["pH_vals"]
    phases    = grid["phases"]        # phases[i_U][i_pH]
    all_phases= grid["all_phases"]
    p_colors  = grid["phase_colors"]

    # Convert phase labels to integer indices for the heatmap
    phase_idx = {p: i for i, p in enumerate(all_phases)}
    z = np.array([[phase_idx[phases[i][j]]
                   for j in range(len(pH_vals))]
                  for i in range(len(U_vals))], dtype=float)

    # Build a discrete colourscale: one colour per phase
    n = len(all_phases)
    colorscale = []
    for i, phase in enumerate(all_phases):
        lo = i / n
        hi = (i + 1) / n
        col = p_colors[phase]
        colorscale.append([lo, col])
        colorscale.append([hi, col])

    fig = go.Figure()

    # ── Heatmap (one cell per grid point) ──
    fig.add_trace(go.Heatmap(
        x=pH_vals,
        y=U_vals,
        z=z,
        colorscale=colorscale,
        zmin=-0.5,
        zmax=n - 0.5,
        showscale=False,
        hovertemplate=(
            "pH = %{x:.1f}<br>"
            "U = %{y:.3f} V vs RHE<br>"
            "<extra></extra>"
        ),
    ))

    # ── Invisible scatter traces for the legend ──
    for phase in all_phases:
        col = p_colors[phase]
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=col, symbol="square"),
            name=phase,
            showlegend=True,
        ))

    # ── Phase boundary contours (optional visual aid) ──
    # Draw thin contour lines at integer transitions in z
    fig.add_trace(go.Contour(
        x=pH_vals,
        y=U_vals,
        z=z,
        contours=dict(
            start=0.5,
            end=n - 0.5,
            size=1.0,
            coloring="none",
        ),
        line=dict(color="rgba(0,0,0,0.4)", width=1),
        showscale=False,
        hoverinfo="skip",
        name="boundaries",
        showlegend=False,
    ))

    # ── SHE reference line: U_RHE = U_SHE - 0.059*pH ──
    she_U = [0.0 - 0.05916 * ph for ph in pH_vals]
    fig.add_trace(go.Scatter(
        x=pH_vals, y=she_U,
        mode="lines",
        name="U = 0 V vs SHE",
        line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
        hoverinfo="skip",
    ))

    axis_style = dict(
        gridcolor=GRID, zerolinecolor=GRID,
        linecolor=MUTED, tickcolor=MUTED,
    )

    fig.update_layout(
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
        title=dict(
            text="Pourbaix Diagram  ·  most stable surface phase",
            font=dict(size=12, color=TEXT),
            x=0.01, xanchor="left",
        ),
        xaxis=dict(title="pH", range=[min(pH_vals), max(pH_vals)], **axis_style),
        yaxis=dict(title="U (V vs RHE)", range=[min(U_vals), max(U_vals)], **axis_style),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=GRID, borderwidth=1,
            font=dict(size=10), title=dict(text="phase", font=dict(size=10)),
        ),
        height=520,
        margin=dict(l=60, r=160, t=44, b=54),
    )

    return fig


# ---------------------------------------------------------------------------
# Combined HTML output
# ---------------------------------------------------------------------------

def build_pourbaix_html(
    db_path: str | Path,
    config_path: str | Path,
    output_path: str | Path = "gocia_pourbaix.html",
    species: str = "O",
    mu_range: tuple[float, float] = (-3.0, 0.0),
    U_range: tuple[float, float] = (-1.0, 1.0),
    pH_range: tuple[float, float] = (0.0, 14.0),
    n_U: int = 120,
    n_pH: int = 100,
    n_mu: int = 400,
    temperature: float | None = None,
    pressure: float | None = None,
) -> Path:
    """
    Load DB + config, compute both diagrams, write a two-panel HTML file.

    Parameters
    ----------
    db_path : str or Path
    config_path : str or Path
        Path to gocia.yaml.
    output_path : str or Path
        Destination HTML file.
    species : str
        Species for the 1-D Δμ sweep.
    mu_range : (float, float)
        Δμ range for the 1-D sweep (eV).
    U_range : (float, float)
        Electrode potential range for the Pourbaix grid (V vs RHE).
    pH_range : (float, float)
        pH range for the Pourbaix grid.
    n_U, n_pH : int
        Pourbaix grid resolution.
    n_mu : int
        Number of points in the 1-D sweep.
    temperature, pressure : float or None
        Override config conditions.

    Returns
    -------
    Path
        The HTML file that was written.
    """
    from datetime import datetime
    import plotly.io as pio

    from gocia.config import load_config
    from gocia.database.db import GociaDB

    cfg = load_config(config_path)

    with GociaDB(db_path) as db:
        df = db.to_dataframe()

    # 1-D sweep
    sweep_df = coverage_vs_mu(
        df, cfg,
        species=species,
        mu_range=mu_range,
        n_points=n_mu,
        temperature=temperature,
        pressure=pressure,
    )
    f_mu = fig_coverage_vs_mu(sweep_df, species=species)

    # 2-D Pourbaix
    grid = pourbaix_grid(
        df, cfg,
        U_range=U_range,
        pH_range=pH_range,
        n_U=n_U,
        n_pH=n_pH,
        temperature=temperature,
        pressure=pressure,
    )
    f_pb = fig_pourbaix(grid)

    # Write a two-panel HTML manually so both figures share one page
    run_label = Path(db_path).parent.resolve().name
    n_structs = len(df[df["status"].isin(["converged","isomer"])])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    mu_html = pio.to_html(f_mu, full_html=False, include_plotlyjs=False,
                          config={"responsive": True})
    pb_html = pio.to_html(f_pb, full_html=False, include_plotlyjs=False,
                          config={"responsive": True})

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GOCIA · Pourbaix · {run_label}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: {BG};
    color: {TEXT};
    font-family: 'IBM Plex Mono', monospace;
    padding: 20px 24px 48px;
  }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 20px;
    border-bottom: 1px solid {GRID};
    padding-bottom: 12px;
  }}
  h1 {{ font-size: 1.05rem; font-weight: 600; color: #e8f0f8; }}
  .meta {{ font-size: 0.63rem; color: {MUTED}; letter-spacing: 0.06em; }}
  .panel {{
    background: {PAPER};
    border: 1px solid {GRID};
    border-radius: 4px;
    margin-bottom: 16px;
    overflow: hidden;
  }}
  .panel-title {{
    font-size: 0.60rem;
    color: {MUTED};
    letter-spacing: 0.10em;
    text-transform: uppercase;
    padding: 7px 12px;
    border-bottom: 1px solid {GRID};
  }}
  .panel-title span {{ color: {TEXT}; }}
</style>
</head>
<body>
<header>
  <h1>GOCIA &nbsp;·&nbsp; Phase Diagrams &nbsp;·&nbsp; {run_label}</h1>
  <span class="meta">{n_structs} converged structures &nbsp;·&nbsp; {timestamp}</span>
</header>

<div class="panel">
  <div class="panel-title">
    Coverage phase diagram &nbsp;·&nbsp;
    <span>most stable phase vs Δμ<sub>{species}</sub>
    &nbsp; (U = {cfg.conditions.potential} V, pH = {cfg.conditions.pH})</span>
  </div>
  {mu_html}
</div>

<div class="panel">
  <div class="panel-title">
    Pourbaix diagram &nbsp;·&nbsp;
    <span>most stable surface phase as a function of U and pH</span>
  </div>
  {pb_html}
</div>

</body>
</html>
"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path
