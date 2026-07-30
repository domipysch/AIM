"""Figure production for the GUI.

Every figure is an interactive Plotly object built here from data read straight
off disk (each K's ``analysis/data`` folder) and, for the UMAP / profile /
fractions / merge-map views, the reference scaffold. The headline UMAP(s) +
spatial map share a single state legend and are recoloured client-side; the
spatial scatter in particular is rebuilt here on every confidence-threshold
change so spots below the threshold can be drawn grey.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from anndata import AnnData
from plotly.subplots import make_subplots

from adata_schema import (
    OBS_COMPUTED_STATE,
    OBS_LEIDEN_ALL_GENES,
    OBSM_UMAP,
    OBSM_UMAP_SHARED_GENES,
    UNS_SHARED_GENES,
)
from analysis.loading import (
    infer_cell_to_state_cluster,
    load_leiden_to_state,
)
from analysis.utils import to_dense

from . import data_access

logger = logging.getLogger(__name__)

# Grey (as a CSS rgba string) for spots drawn below the confidence threshold.
_GREY_RGBA = "rgba(184,184,184,0.9)"

# Matplotlib's tab20 / tab10 qualitative palettes as hex strings, inlined so the
# GUI carries no matplotlib dependency. tab20 keys the per-state colours (state
# id -> colour); tab10 distinguishes mappers in the Compare tab and the K-sweep
# lines. Values match matplotlib exactly, so figures look identical to before.
_TAB20 = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]
_TAB10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# Shared Plotly look used by every figure: a white template, transparent
# backgrounds so it blends with the app, and the app's sans-serif font.
_FONT = dict(family="'Source Sans Pro', 'Segoe UI', sans-serif", size=13)
# The horizontal state-legend strip shared by the headline / compare / reference
# figures; each sets its own vertical ``y`` offset via ``{**_STATE_LEGEND, ...}``.
_STATE_LEGEND = dict(
    itemsizing="constant",
    title="States",
    orientation="h",
    yanchor="bottom",
    xanchor="left",
    x=0,
)


def _base_layout(
    fig: go.Figure, *, height: int, font: dict = _FONT, **extra
) -> go.Figure:
    """Apply the shared template/background/font to ``fig`` (returns ``fig``).

    Every other layout knob (legend, margin, title, axis titles, barmode, …) is
    passed through ``extra`` unchanged.
    """
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=font,
        height=height,
        **extra,
    )
    return fig


def figure_to_bytes(fig: go.Figure, fmt: str = "png", *, scale: float = 2.0) -> bytes:
    """Static-export a Plotly figure to PNG / SVG / PDF bytes via kaleido.

    ``scale`` up-samples raster (png) output for crisp exports; it is ignored by
    the vector formats. Raises ``RuntimeError`` with an install hint if kaleido
    is missing. Note: the UMAP / spatial panels use WebGL (``Scattergl``) traces,
    which kaleido rasterises inside svg/pdf — png is always faithful.
    """
    try:
        return fig.to_image(format=fmt, scale=scale)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        raise RuntimeError(
            f"Could not export as {fmt.upper()}: {exc}. "
            "Static export needs the 'kaleido' package (pip install kaleido)."
        ) from exc


def state_palette(k: int) -> dict[int, str]:
    """tab20 palette keyed by state id, as ``#rrggbb`` hex strings."""
    return {s: _TAB20[s % 20] for s in range(k)}


def _hex(color: str | None) -> str:
    """Pass a hex colour through, with a grey fallback for a missing (None) state."""
    return color if color else "#b8b8b8"


def _mod_suffix(mod: dict | None, key: str) -> str:
    """`` (modularity = 0.123)`` for modularity ``key`` in ``mod`` (else ``''``),
    appended to a subplot title so each UMAP shows its own modularity."""
    v = (mod or {}).get(key)
    return f"  (modularity = {v:.3f})" if isinstance(v, (int, float)) else ""


def _padded_range(vals: np.ndarray, frac: float = 0.03) -> tuple[float, float]:
    """``(lo, hi)`` covering all of ``vals`` with a small margin.

    Used to pin axis ranges from the full data, so hiding states via the legend
    never re-autoranges (zooms) the view.
    """
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = hi - lo
    pad = frac * span if span > 0 else 1.0
    return lo - pad, hi + pad


def _style_scatter_axes(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    xref: str,
    coords: np.ndarray,
    xtitle: str,
    ytitle: str,
    equal_aspect: bool,
    reversed_y: bool = False,
    hide_ticks: bool = False,
) -> None:
    """Set fixed axis ranges for the subplot at (``row``, ``col``) (so state
    recolour/toggle never re-zooms), optionally flipping y and hiding tick labels.

    ``equal_aspect`` adds ``scaleanchor``/``constrain='domain'`` for a true 1:1
    ratio. ``constrain='domain'`` only ever *shrinks* a panel within its own
    subplot cell (centring it with whitespace), so it never collides and is safe
    for any grid. ``xref`` is this subplot's x-axis id (e.g. "x", "x2") that the
    equal-aspect ``scaleanchor`` must reference — taken from a trace already added
    to the subplot, so it stays correct in any multi-row/-column layout.
    """
    xr = _padded_range(coords[:, 0])
    yr = _padded_range(coords[:, 1])
    xkw = dict(title_text=xtitle, range=list(xr), autorange=False, row=row, col=col)
    ykw = dict(
        title_text=ytitle,
        range=[yr[1], yr[0]] if reversed_y else list(yr),
        autorange=False,
        row=row,
        col=col,
    )
    if hide_ticks:
        xkw["showticklabels"] = False
        ykw["showticklabels"] = False
    if equal_aspect:
        xkw["constrain"] = "domain"
        ykw["constrain"] = "domain"
        ykw["scaleanchor"] = xref
        ykw["scaleratio"] = 1
    fig.update_xaxes(**xkw)
    fig.update_yaxes(**ykw)


def _add_umap_traces(
    fig: go.Figure,
    adata_sc: AnnData,
    root: Path,
    k: int,
    *,
    col: int,
    row: int = 1,
    palette: dict[int, tuple],
    legend_shown: set[int],
    dot_size: float,
    umap_key: str = OBSM_UMAP,
    equal_aspect: bool = True,
) -> None:
    """Add a UMAP scatter (one trace per state) to subplot (``row``, ``col``).

    ``umap_key`` selects the embedding (all-gene ``X_umap`` by default, or the
    shared-gene ``X_umap_shared_genes``). Applies this K's tree cut to the
    scaffold (``infer_cell_to_state_cluster``) so ``computed_state`` is correct
    for ``k`` regardless of PNG-cache state. Each trace joins ``legendgroup``
    ``state<n>`` so the interaction layer can toggle the same state in every
    subplot; a state gets a legend entry only the first time it is seen (tracked
    in ``legend_shown``).
    """
    leiden_to_state = load_leiden_to_state(data_access.k_dir(root, k))
    infer_cell_to_state_cluster(adata_sc, leiden_to_state)

    coords = np.asarray(adata_sc.obsm[umap_key])
    states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()
    leiden = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()

    for state in sorted(np.unique(states).tolist()):
        mask = states == state
        show = state not in legend_shown
        legend_shown.add(state)
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(size=dot_size, color=_hex(palette.get(state)), opacity=1.0),
                name=f"State {state}",
                legendgroup=f"state{state}",
                showlegend=show,
                customdata=leiden[mask][:, None],
                hovertemplate=(
                    f"State {state}<br>Leiden subcluster %{{customdata[0]}}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    _style_scatter_axes(
        fig,
        row=row,
        col=col,
        xref=fig.data[-1].xaxis or "x",
        coords=coords,
        xtitle="UMAP1",
        ytitle="UMAP2",
        equal_aspect=equal_aspect,
        hide_ticks=True,
    )


def _pin_spatial_axes(
    fig: go.Figure, coords: np.ndarray, *, col: int, row: int = 1, equal_aspect: bool
) -> None:
    """Style the spatial subplot at (``row``, ``col``): fixed extent, flipped y,
    and equal aspect only when ``equal_aspect`` (see ``_style_scatter_axes``).
    ``xref`` is read from the last-added trace, which belongs to this subplot."""
    _style_scatter_axes(
        fig,
        row=row,
        col=col,
        xref=fig.data[-1].xaxis or "x",
        coords=coords,
        xtitle="x",
        ytitle="y",
        equal_aspect=equal_aspect,
        reversed_y=True,
    )


def _add_spatial_traces(
    fig: go.Figure,
    coords: np.ndarray,
    hard: np.ndarray,
    confidence: np.ndarray | None,
    threshold: float,
    *,
    col: int,
    row: int = 1,
    palette: dict[int, tuple],
    legend_shown: set[int],
    dot_size: float,
    plot_confidence: bool = False,
    equal_aspect: bool = True,
) -> None:
    """Add the ST spatial scatter to subplot (``row``, ``col``).

    Default: one trace per state (colours from ``palette``); spots below
    ``threshold`` are drawn grey. State traces share the ``state<n>`` legendgroup
    with the UMAP so one legend controls both panels.

    When ``plot_confidence`` is set and confidence exists, instead draw a single
    trace coloured by a continuous confidence colourscale (with a colourbar); the
    threshold is ignored in this mode.
    """
    if plot_confidence and confidence is not None:
        fig.add_trace(
            go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker=dict(
                    size=dot_size,
                    color=confidence,
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=1.0,
                    showscale=True,
                    colorbar=dict(title="confidence", thickness=12, len=0.85, x=1.0),
                ),
                name="confidence",
                showlegend=False,
                meta="spatial",
                customdata=np.column_stack([confidence, hard]),
                hovertemplate=(
                    "confidence %{customdata[0]:.3f}<br>"
                    "state %{customdata[1]}<br>"
                    "x %{x:.1f}  y %{y:.1f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        _pin_spatial_axes(fig, coords, row=row, col=col, equal_aspect=equal_aspect)
        return

    if confidence is not None and threshold > 0.0:
        low = confidence < threshold
    else:
        low = np.zeros(len(hard), dtype=bool)
    keep = ~low

    n_low = int(low.sum())
    if n_low:
        # Only reached when confidence is not None (threshold > 0 required above).
        fig.add_trace(
            go.Scattergl(
                x=coords[low, 0],
                y=coords[low, 1],
                mode="markers",
                marker=dict(size=dot_size, color=_GREY_RGBA),
                name=f"below threshold ({n_low})",
                legendgroup="below_threshold",
                meta="spatial",
                customdata=np.column_stack([hard[low], confidence[low]]),
                hovertemplate=(
                    "State %{customdata[0]} (below)<br>"
                    "confidence %{customdata[1]:.3f}<br>"
                    "x %{x:.1f}  y %{y:.1f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    for state in sorted(np.unique(hard[keep]).tolist()):
        mask = keep & (hard == state)
        show = state not in legend_shown
        legend_shown.add(state)
        if confidence is not None:
            customdata = confidence[mask][:, None]
            hovertemplate = (
                f"State {state}<br>"
                "confidence %{customdata[0]:.3f}<br>"
                "x %{x:.1f}  y %{y:.1f}<extra></extra>"
            )
        else:
            customdata = None
            hovertemplate = (
                f"State {state}<br>x %{{x:.1f}}  y %{{y:.1f}}<extra></extra>"
            )
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(size=dot_size, color=_hex(palette.get(state)), opacity=1.0),
                name=f"State {state}",
                legendgroup=f"state{state}",
                showlegend=show,
                meta="spatial",
                customdata=customdata,
                hovertemplate=hovertemplate,
            ),
            row=row,
            col=col,
        )

    _pin_spatial_axes(fig, coords, row=row, col=col, equal_aspect=equal_aspect)


def render_headline_figure(
    coords: np.ndarray | None,
    hard: np.ndarray,
    confidence: np.ndarray | None,
    threshold: float,
    k: int,
    *,
    adata_sc: AnnData | None = None,
    root: Path | None = None,
    plot_confidence: bool = False,
    show_shared_umap: bool = False,
    dot_size_umap: float = 4.0,
    dot_size_spatial: float = 6.0,
) -> go.Figure:
    """One interactive figure holding the UMAP(s) and the spatial map as
    side-by-side subplots that share a single state legend.

    The reference (all-gene) UMAP is included when ``adata_sc`` (and ``root``)
    are given; with ``show_shared_umap`` a second, shared-gene UMAP is added to
    its left. The spatial panel is included when ``coords`` is not None. Every
    panel colours states from one ``state_palette(k)`` keyed by state id, and a
    state's traces share a ``legendgroup`` so a single click toggles it
    everywhere.

    With ``plot_confidence`` the spatial panel is coloured by a continuous
    confidence scale (with a colourbar) instead of by state.
    """
    palette = state_palette(k)
    have_umap = adata_sc is not None and root is not None
    have_shared = (
        have_umap and show_shared_umap and OBSM_UMAP_SHARED_GENES in adata_sc.obsm
    )
    have_spatial = coords is not None
    conf_mode = plot_confidence and have_spatial and confidence is not None

    mod = (
        data_access.load_data_json(root, k, "modularity_metrics.json")
        if root is not None
        else None
    )
    titles: list[str] = []
    if have_shared:
        titles.append(
            "Shared-gene UMAP — computed states" + _mod_suffix(mod, "modularity_shared")
        )
    if have_umap:
        titles.append(
            "Reference UMAP — computed states" + _mod_suffix(mod, "modularity_all")
        )
    if have_spatial:
        if conf_mode:
            spatial_title = "Spatial confidence"
        else:
            spatial_title = "Spatial cell states"
            if confidence is not None and threshold > 0.0:
                spatial_title += f" (confidence ≥ {threshold:.2f})"
        titles.append(spatial_title)

    n_cols = max(1, len(titles))
    spacing = 0.06 if n_cols >= 3 else 0.08
    fig = make_subplots(
        rows=1, cols=n_cols, subplot_titles=titles, horizontal_spacing=spacing
    )

    # Equal aspect (scaleanchor + constrain="domain") gives every panel a true
    # 1:1 ratio by shrinking it within its own subplot cell, centred with
    # whitespace. Because it only shrinks the domain it never overlaps neighbours,
    # so it is applied uniformly for any column count (the client no longer
    # post-corrects the aspect ratio).
    equal_aspect = True

    legend_shown: set[int] = set()
    col = 1
    # Each adder pins its own subplot axes.
    if have_shared:
        _add_umap_traces(
            fig,
            adata_sc,
            root,
            k,
            col=col,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_umap,
            umap_key=OBSM_UMAP_SHARED_GENES,
            equal_aspect=equal_aspect,
        )
        col += 1
    if have_umap:
        _add_umap_traces(
            fig,
            adata_sc,
            root,
            k,
            col=col,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_umap,
            umap_key=OBSM_UMAP,
            equal_aspect=equal_aspect,
        )
        col += 1
    if have_spatial:
        _add_spatial_traces(
            fig,
            coords,
            hard,
            confidence,
            threshold,
            col=col,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_spatial,
            plot_confidence=plot_confidence,
            equal_aspect=equal_aspect,
        )

    # Keep the state legend as a fixed horizontal strip on top in every mode, so
    # it never jumps when the confidence colourbar appears on the right. The
    # ``y`` offset leaves a little vertical gap between the legend and the plots.
    return _base_layout(
        fig,
        height=720,
        legend={**_STATE_LEGEND, "y": 1.12},
        margin=dict(l=10, r=10, t=70, b=10),
    )


def render_compare_figure(
    coords: np.ndarray,
    hards: list[np.ndarray],
    mapper_names: list[str],
    k: int,
    *,
    adata_sc: AnnData | None = None,
    root: Path | None = None,
    dot_size_umap: float = 4.0,
    dot_size_spatial: float = 6.0,
) -> go.Figure:
    """One reference UMAP centred on top, then the selected mappers' spatial maps
    in a grid below (two per row), all in one figure with a single shared state
    legend — so one legend click toggles a state across every panel at once.

    The spatial coordinates are shared across mappers; only each mapper's hard
    assignment (``hards[i]`` for ``mapper_names[i]``) differs. No shared-gene UMAP
    and no confidence colouring here.
    """
    palette = state_palette(k)
    have_umap = adata_sc is not None and root is not None
    n = len(mapper_names)
    n_spatial_rows = (n + 1) // 2

    # Grid: (optional) row 1 = UMAP spanning both columns so equal-aspect centres
    # it with whitespace; each later row holds up to two spatial panels.
    specs: list[list] = []
    if have_umap:
        specs.append([{"colspan": 2}, None])
    for r in range(n_spatial_rows):
        specs.append([{}, {}] if n - r * 2 >= 2 else [{}, None])
    n_rows = len(specs)

    titles: list[str] = []
    if have_umap:
        titles.append("Reference UMAP — computed states")
    titles += [f"Spatial — {m}" for m in mapper_names]

    # Per-row pixel budgets (row_heights normalises these into proportions): the
    # spatial rows are taller than the UMAP row so the maps get more room.
    umap_px = 420
    spatial_px = 640
    row_heights = ([umap_px] if have_umap else []) + [spatial_px] * n_spatial_rows
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        specs=specs,
        subplot_titles=titles,
        row_heights=row_heights,
        horizontal_spacing=0.06,
        vertical_spacing=(min(0.1, 0.8 / (n_rows - 1)) if n_rows > 1 else 0.0),
    )

    legend_shown: set[int] = set()
    umap_row = 1 if have_umap else 0
    if have_umap:
        _add_umap_traces(
            fig,
            adata_sc,
            root,
            k,
            row=1,
            col=1,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_umap,
            umap_key=OBSM_UMAP,
        )
    for j, hard in enumerate(hards):
        _add_spatial_traces(
            fig,
            coords,
            hard,
            None,
            0.0,
            row=umap_row + 1 + j // 2,
            col=1 + j % 2,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_spatial,
            plot_confidence=False,
        )

    height = (umap_px if have_umap else 0) + spatial_px * n_spatial_rows + 60
    return _base_layout(
        fig,
        height=height,
        legend={**_STATE_LEGEND, "y": 1.02},
        margin=dict(l=10, r=10, t=60, b=10),
    )


# --------------------------------------------------------------------------- #
# Compare-tab combined report cards (all selected mappers in one plot/table)
# --------------------------------------------------------------------------- #
def _mapper_color(i: int) -> str:
    """Stable qualitative colour for the i-th mapper in a comparison."""
    return _TAB10[i % 10]


def render_compare_reconstruction_figure(cossims: dict[str, dict]) -> go.Figure | None:
    """Two-panel (gene-wise, spot-wise) box plot of reconstruction cosine
    similarity, with every selected mapper's two hard x raw/norm combos in the
    same axes (grouped by combo, coloured by mapper).

    ``cossims`` maps each mapper to ``{combo: {"per_gene": [...], "per_spot": [...]}}``.
    Returns ``None`` if no data is present.
    """
    order = ["hard-raw", "hard-norm"]
    mappers = [m for m in cossims if cossims[m]]
    if not mappers:
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Gene-wise", "Spot-wise"],
        horizontal_spacing=0.12,
    )
    for col, attr in ((1, "per_gene"), (2, "per_spot")):
        for i, m in enumerate(mappers):
            xs: list[str] = []
            ys: list[float] = []
            for combo in order:
                vals = (cossims[m].get(combo) or {}).get(attr) or []
                xs += [combo] * len(vals)
                ys += list(vals)
            if not ys:
                continue
            fig.add_trace(
                go.Box(
                    x=xs,
                    y=ys,
                    name=m,
                    legendgroup=m,
                    showlegend=(col == 1),
                    marker_color=_mapper_color(i),
                    boxpoints=False,
                ),
                row=1,
                col=col,
            )
    _base_layout(
        fig,
        height=340,
        font=_CARD_FONT,
        boxmode="group",
        margin=dict(l=10, r=10, t=30, b=52),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
    )
    fig.update_yaxes(title_text="cosine similarity", row=1, col=1)
    fig.update_xaxes(tickangle=-30)
    return fig


def render_compare_box_figure(
    data: dict[str, np.ndarray],
    *,
    title: str,
    ytitle: str,
    yrange: tuple[float, float] | None = None,
    height: int = 300,
) -> go.Figure:
    """One box per mapper of the per-spot distribution in ``data`` (mapper ->
    values). Used to compare mapping sharpness (max probability) and confidence."""
    fig = go.Figure()
    for i, (m, vals) in enumerate(data.items()):
        fig.add_trace(
            go.Box(
                y=np.asarray(vals, dtype=float),
                name=m,
                marker_color=_mapper_color(i),
                boxpoints=False,
                showlegend=False,
            )
        )
    _base_layout(
        fig,
        height=height,
        font=_CARD_FONT,
        title=dict(text=title, font=dict(size=13)),
        yaxis_title=ytitle,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    if yrange is not None:
        fig.update_yaxes(range=list(yrange))
    return fig


def render_compare_fractions_figure(
    spot_fracs: dict[str, list[float]], k: int, *, height: int = 300
) -> go.Figure:
    """Grouped bar of the spot-state fraction per state, one bar group per state
    and one bar per mapper. (Cell fractions are identical across mappers, so only
    the mapping-dependent spot fractions are compared here.)"""
    states = list(range(k))
    x = [f"State {s}" for s in states]
    fig = go.Figure()
    for i, (m, fr) in enumerate(spot_fracs.items()):
        fig.add_trace(go.Bar(name=m, x=x, y=fr, marker_color=_mapper_color(i)))
    _base_layout(
        fig,
        height=height,
        font=_CARD_FONT,
        barmode="group",
        title=dict(text="Spot-state fractions", font=dict(size=13)),
        yaxis_title="fraction",
        margin=dict(l=10, r=10, t=40, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


# --------------------------------------------------------------------------- #
# Single-cell reference tab (clustering-side, mapper-independent) figures
# --------------------------------------------------------------------------- #
def _add_leiden_umap_traces(
    fig: go.Figure,
    adata_sc: AnnData,
    leiden_to_state: np.ndarray,
    *,
    col: int,
    row: int = 1,
    dot_size: float,
    equal_aspect: bool = True,
) -> None:
    """Add the Leiden-overclustering UMAP (one trace per Leiden subcluster) to
    subplot (``row``, ``col``).

    Each subcluster gets a distinct qualitative colour but is tagged with the
    ``state<n>`` legendgroup of the state it merges into, so a single legend click
    greys its cells here too; the subclusters keep out of the legend (identified
    via hover) to avoid duplicating the shared state legend.
    """
    coords = np.asarray(adata_sc.obsm[OBSM_UMAP])
    leiden = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()
    for i, lc in enumerate(sorted(np.unique(leiden).tolist())):
        mask = leiden == lc
        s = int(leiden_to_state[lc])
        fig.add_trace(
            go.Scattergl(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                marker=dict(size=dot_size, color=_TAB20[i % 20], opacity=1.0),
                name=f"Leiden {lc}",
                legendgroup=f"state{s}",
                showlegend=False,
                hovertemplate=(f"Leiden subcluster {lc}<br>→ State {s}<extra></extra>"),
            ),
            row=row,
            col=col,
        )
    _style_scatter_axes(
        fig,
        row=row,
        col=col,
        xref=fig.data[-1].xaxis or "x",
        coords=coords,
        xtitle="UMAP1",
        ytitle="UMAP2",
        equal_aspect=equal_aspect,
        hide_ticks=True,
    )


def render_reference_umaps_figure(
    adata_sc: AnnData,
    root: Path,
    k: int,
    *,
    dot_size_umap: float = 4.0,
) -> go.Figure:
    """Three reference UMAPs sharing one state legend: the Leiden overclustering
    (left), the computed states on the all-gene UMAP (middle), and the computed
    states on the shared-gene UMAP (right, dropped if that embedding is absent).

    Rendered by the same client-side component as the headline plot, so one
    legend click toggles a state across all three panels (the Leiden subclusters
    grey out with the state they merged into).
    """
    palette = state_palette(k)
    leiden_to_state = load_leiden_to_state(data_access.k_dir(root, k))
    infer_cell_to_state_cluster(adata_sc, leiden_to_state)
    have_shared = OBSM_UMAP_SHARED_GENES in adata_sc.obsm

    mod = data_access.load_data_json(root, k, "modularity_metrics.json")
    titles = [
        "Leiden overclustering",
        "Computed states — reference UMAP" + _mod_suffix(mod, "modularity_all"),
    ]
    if have_shared:
        titles.append(
            "Computed states — shared-gene UMAP" + _mod_suffix(mod, "modularity_shared")
        )
    n_cols = len(titles)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=0.06 if n_cols >= 3 else 0.08,
    )

    legend_shown: set[int] = set()
    _add_leiden_umap_traces(
        fig, adata_sc, leiden_to_state, col=1, dot_size=dot_size_umap
    )
    _add_umap_traces(
        fig,
        adata_sc,
        root,
        k,
        col=2,
        palette=palette,
        legend_shown=legend_shown,
        dot_size=dot_size_umap,
        umap_key=OBSM_UMAP,
    )
    if have_shared:
        _add_umap_traces(
            fig,
            adata_sc,
            root,
            k,
            col=3,
            palette=palette,
            legend_shown=legend_shown,
            dot_size=dot_size_umap,
            umap_key=OBSM_UMAP_SHARED_GENES,
        )

    return _base_layout(
        fig,
        height=520,
        legend={**_STATE_LEGEND, "y": 1.12},
        margin=dict(l=10, r=10, t=70, b=10),
    )


def render_leiden_merge_figure(adata_sc: AnnData, root: Path, k: int) -> go.Figure:
    """Horizontal stacked bars: one bar per computed state, segmented by the
    Leiden subclusters merged into it (segment width ∝ cell count, labelled with
    the Leiden id)."""
    leiden_to_state = load_leiden_to_state(data_access.k_dir(root, k))
    infer_cell_to_state_cluster(adata_sc, leiden_to_state)
    leiden = adata_sc.obs[OBS_LEIDEN_ALL_GENES].astype(int).to_numpy()
    cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()
    palette = state_palette(k)
    states = sorted(np.unique(cell_states).tolist())

    # cell_states is constant within a Leiden cluster, so each maps to one state.
    leiden_of_state: dict[int, list[tuple[int, int]]] = {s: [] for s in states}
    for lc in np.unique(leiden):
        mask = leiden == lc
        s = int(cell_states[mask][0])
        leiden_of_state[s].append((int(lc), int(mask.sum())))
    for s in states:
        leiden_of_state[s].sort(key=lambda t: t[1], reverse=True)

    def _row(s: int) -> str:
        n = len(leiden_of_state[s])
        return f"State {s}  ({n} cluster{'s' if n != 1 else ''})"

    fig = go.Figure()
    # barmode="stack" accumulates same-row segments in trace order (largest first).
    for s in states:
        color = _hex(palette.get(s))
        for lc, size in leiden_of_state[s]:
            fig.add_trace(
                go.Bar(
                    y=[_row(s)],
                    x=[size],
                    orientation="h",
                    marker=dict(color=color, line=dict(color="white", width=1.5)),
                    text=[f"L{lc}"],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=9, color="black"),
                    showlegend=False,
                    hovertemplate=f"State {s} · Leiden {lc}<br>%{{x}} cells<extra></extra>",
                )
            )

    n_states = len(states)
    _base_layout(
        fig,
        height=max(240, n_states * 46 + 120),
        barmode="stack",
        title="Leiden overclusters merged per computed AIM state",
        xaxis_title="cells   (segment width ∝ Leiden subcluster size)",
        margin=dict(l=10, r=10, t=50, b=40),
        bargap=0.35,
    )
    # State 0 on top (inverted y-axis).
    fig.update_yaxes(
        categoryorder="array", categoryarray=[_row(s) for s in reversed(states)]
    )
    return fig


def render_state_profiles_figure(
    adata_sc: AnnData, root: Path, k: int
) -> go.Figure | None:
    """Heatmap of per-state mean expression over the shared genes (sorted by SC
    variance, z-scored per gene across states). Returns ``None`` when too few
    shared genes are present to plot.
    """
    leiden_to_state = load_leiden_to_state(data_access.k_dir(root, k))
    infer_cell_to_state_cluster(adata_sc, leiden_to_state)
    cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()

    shared_genes = list(adata_sc.uns.get(UNS_SHARED_GENES, []))
    available = [g for g in shared_genes if g in adata_sc.var_names]
    if len(available) < 2:
        return None

    X = to_dense(adata_sc[:, available])
    gene_names = np.array(available)
    gene_order = np.argsort(X.var(axis=0))[::-1]

    unique_states = sorted(np.unique(cell_states).tolist())
    mat = np.stack(
        [X[cell_states == s][:, gene_order].mean(axis=0) for s in unique_states]
    )
    col_std = mat.std(axis=0)
    col_std[col_std == 0] = 1.0
    mat_z = (mat - mat.mean(axis=0)) / col_std

    fig = go.Figure(
        go.Heatmap(
            z=mat_z,
            x=gene_names[gene_order].tolist(),
            y=[f"State {s}" for s in unique_states],
            colorscale="Viridis",
            colorbar=dict(title="z-score", thickness=12),
            hovertemplate="%{y}<br>%{x}<br>z %{z:.2f}<extra></extra>",
        )
    )
    n_states = len(unique_states)
    _base_layout(
        fig,
        height=max(240, n_states * 42 + 160),
        title="Cell-state profiles — shared genes (z-scored per gene across states)",
        xaxis_title="Gene  (sorted by SC variance)",
        yaxis_title="Computed cell state",
        margin=dict(l=10, r=10, t=50, b=80),
    )
    # State 0 on top.
    fig.update_yaxes(autorange="reversed")
    return fig


# --------------------------------------------------------------------------- #
# Per-mapper report dashboard figures (interactive Plotly cards)
# --------------------------------------------------------------------------- #
_CARD_FONT = dict(family="'Source Sans Pro', 'Segoe UI', sans-serif", size=12)
_CELL_COLOR = "#4c78a8"  # cell / soft
_SPOT_COLOR = "#f58518"  # spot / hard


def _card_layout(fig: go.Figure, *, height: int, **extra) -> go.Figure:
    """Apply the shared compact look used by every report-dashboard card."""
    extra.setdefault("margin", dict(l=10, r=10, t=30, b=10))
    return _base_layout(fig, height=height, font=_CARD_FONT, **extra)


def _ksweep_line_figure(
    df,
    series: list[tuple[str, str]],
    *,
    title: str,
    ytitle: str,
    height: int = 300,
    colors: list[str] | None = None,
) -> go.Figure:
    """A K-sweep line chart: each ``(column, label)`` in ``series`` becomes a
    line over K. Shared look for the three per-method K-sweep cards. ``colors``,
    when given, sets an explicit per-series colour (by position in ``series``) so
    lines stay distinct across the separate cards."""
    fig = go.Figure()
    for i, (col, label) in enumerate(series):
        if col in df.columns:
            c = colors[i] if colors else None
            fig.add_trace(
                go.Scatter(
                    x=df["k"],
                    y=df[col],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=c),
                    marker=dict(color=c),
                    hovertemplate=f"{label}<br>K %{{x}}<br>%{{y:.3f}}<extra></extra>",
                )
            )
    return _base_layout(
        fig,
        height=height,
        font=_CARD_FONT,
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="K",
        yaxis_title=ytitle,
        margin=dict(l=10, r=10, t=40, b=52),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
    )


def render_ksweep_reconstruction_figure(df) -> go.Figure:
    """K-sweep reconstruction cosine similarity: the raw hard-assignment combos
    (spot/gene) on one y-axis over K."""
    return _ksweep_line_figure(
        df,
        [
            ("cossim_hard_raw_spot", "raw · spot"),
            ("cossim_hard_raw_gene", "raw · gene"),
        ],
        title="Reconstruction cosine similarity",
        ytitle="cosine similarity",
        colors=[_mapper_color(0), _mapper_color(1)],
    )


def render_ksweep_spatial_figure(df, *, height: int = 300) -> go.Figure:
    """K-sweep spatial organisation: neighbourhood-enrichment and local-purity
    z-scores over K, each on its own y-axis (nhood left, local purity right)
    because the two z-scores live on very different scales."""
    palette = [_mapper_color(2), _mapper_color(3)]
    fig = go.Figure()
    if "nhood_mean_self_zscore" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["k"],
                y=df["nhood_mean_self_zscore"],
                mode="lines+markers",
                name="nhood enrichment z",
                yaxis="y",
                line=dict(color=palette[0]),
                marker=dict(color=palette[0]),
                hovertemplate="nhood enrichment z<br>K %{x}<br>%{y:.3f}<extra></extra>",
            )
        )
    if "local_purity_zscore" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["k"],
                y=df["local_purity_zscore"],
                mode="lines+markers",
                name="local purity z",
                yaxis="y2",
                line=dict(color=palette[1]),
                marker=dict(color=palette[1]),
                hovertemplate="local purity z<br>K %{x}<br>%{y:.3f}<extra></extra>",
            )
        )
    return _base_layout(
        fig,
        height=height,
        font=_CARD_FONT,
        title=dict(text="Spatial organisation", font=dict(size=13)),
        xaxis_title="K",
        yaxis=dict(
            title=dict(text="nhood enrichment z", font=dict(color=palette[0])),
            tickfont=dict(color=palette[0]),
        ),
        yaxis2=dict(
            title=dict(text="local purity z", font=dict(color=palette[1])),
            tickfont=dict(color=palette[1]),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        margin=dict(l=10, r=10, t=40, b=52),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
    )


def render_ksweep_coherence_figure(df) -> go.Figure:
    """K-sweep transcriptional coherence over K: the ST-expression modularity of
    the mapped spots (mapping-side) alongside the reference-side modularity of the
    computed-state partition on the sc shared-gene graph (mapper-independent)."""
    return _ksweep_line_figure(
        df,
        [
            ("modularity_st_expression", "ST expression (mapping)"),
            ("modularity_shared", "SC shared-gene (reference)"),
        ],
        title="Transcriptional coherence",
        ytitle="modularity",
        colors=[_mapper_color(4), _mapper_color(5)],
    )


def render_fractions_figure(
    adata_sc: AnnData, root: Path, k: int, hard: np.ndarray
) -> go.Figure:
    """Cell fraction (reference states) and spot fraction (this mapper's
    assignment) per computed state, as one grouped-bar plot sharing a single
    y-axis. Both bars use the same per-state palette as the UMAP/spatial plots;
    the spot bars are hatched to tell them apart."""
    leiden_to_state = load_leiden_to_state(data_access.k_dir(root, k))
    infer_cell_to_state_cluster(adata_sc, leiden_to_state)
    cell_states = adata_sc.obs[OBS_COMPUTED_STATE].astype(int).to_numpy()
    hard = np.asarray(hard).astype(int)

    palette = state_palette(k)
    states = list(range(k))
    x = [f"State {s}" for s in states]
    colors = [_hex(palette.get(s)) for s in states]
    cell_frac = [float(np.mean(cell_states == s)) for s in states]
    spot_frac = [float(np.mean(hard == s)) for s in states]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Cell (solid)",
            x=x,
            y=cell_frac,
            marker=dict(color=colors),
            hovertemplate="Cell · %{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Spot (hatched)",
            x=x,
            y=spot_frac,
            marker=dict(
                color=colors,
                pattern=dict(shape="/", size=6, solidity=0.55, fgcolor="white"),
            ),
            hovertemplate="Spot · %{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    _card_layout(
        fig,
        height=320,
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        yaxis_title="fraction",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def render_reconstruction_figure(cossim: dict[str, dict]) -> go.Figure | None:
    """Two-panel (gene-wise, spot-wise) box plot of reconstruction cosine
    similarity for the soft/hard x raw/norm combos. Returns ``None`` if no combos
    are present.

    ``cossim`` maps each label to ``{"per_gene": [...], "per_spot": [...]}``.
    """
    order = ["soft-raw", "hard-raw", "soft-norm", "hard-norm"]
    labels = [lbl for lbl in order if lbl in cossim]
    if not labels:
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Gene-wise", "Spot-wise"],
        horizontal_spacing=0.12,
    )
    for col, attr in ((1, "per_gene"), (2, "per_spot")):
        for lbl in labels:
            fig.add_trace(
                go.Box(
                    y=cossim[lbl][attr],
                    name=lbl,
                    marker_color=_CELL_COLOR if lbl.startswith("soft") else _SPOT_COLOR,
                    boxpoints=False,
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
    _card_layout(fig, height=320)
    fig.update_yaxes(title_text="cosine similarity", row=1, col=1)
    fig.update_xaxes(tickangle=-30)
    return fig


def render_onehot_figure(max_prob: np.ndarray) -> go.Figure:
    """Histogram of per-spot max probability (1.0 = fully one-hot), with mean and
    median markers."""
    max_prob = np.asarray(max_prob, dtype=float)
    mean, median = float(np.mean(max_prob)), float(np.median(max_prob))

    fig = go.Figure(go.Histogram(x=max_prob, nbinsx=40, marker_color=_CELL_COLOR))
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color="black",
        annotation_text=f"mean {mean:.3f}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=median,
        line_dash="dot",
        line_color=_SPOT_COLOR,
        annotation_text=f"median {median:.3f}",
        annotation_position="top right",
    )
    _card_layout(
        fig,
        height=300,
        xaxis_title="max probability per spot (1.0 = one-hot)",
        yaxis_title="spots",
        bargap=0.02,
    )
    return fig


def render_confidence_figure(confidence: np.ndarray) -> go.Figure:
    """Histogram of per-spot assignment confidence in [0, 1], with mean and
    median markers."""
    confidence = np.asarray(confidence, dtype=float)
    mean, median = float(np.mean(confidence)), float(np.median(confidence))

    fig = go.Figure(
        go.Histogram(
            x=confidence,
            xbins=dict(start=0.0, end=1.0, size=1.0 / 40),
            marker_color="#2ca02c",
        )
    )
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color="black",
        annotation_text=f"mean {mean:.3f}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=median,
        line_dash="dot",
        line_color=_SPOT_COLOR,
        annotation_text=f"median {median:.3f}",
        annotation_position="top right",
    )
    _card_layout(
        fig,
        height=300,
        xaxis_title="assignment confidence per spot",
        yaxis_title="spots",
        bargap=0.02,
    )
    fig.update_xaxes(range=[0.0, 1.0])
    return fig
