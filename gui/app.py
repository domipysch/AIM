"""Streamlit UI for browsing AIM sweep results.

Launched by ``gui/__main__.py`` via ``streamlit run gui/app.py -- <args>``. Not
meant to be run directly. Reads the up-front CLI args (sc/ST/output/K-range),
lets the user run one mapper at a time, then browses each mapper's per-K results
with a K slider, a live confidence-threshold slider, the UMAP + spatial plots on
top, the report sections below, and the K-sweep plot -- with a Compare tab for
two mappers side by side.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple

# Ensure the in-repo packages under src/ are importable even if PYTHONPATH was
# not inherited (e.g. when Streamlit re-execs the script).
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from adata_schema import OBSM_UMAP_SHARED_GENES  # noqa: E402
from aim import MAPPING_CHOICES  # noqa: E402

from gui import compute, data_access, render, scaffold, widgets  # noqa: E402

if TYPE_CHECKING:
    from anndata import AnnData

# --------------------------------------------------------------------------- #
# Args & cached loaders
# --------------------------------------------------------------------------- #
# K range: min/max are always the defaults (full sweep); only the step is
# user-editable in the sidebar.
_DEFAULT_K_MIN: int | None = None
_DEFAULT_K_MAX: int | None = None
_DEFAULT_K_STEP = 1


@st.cache_resource
def _cli_defaults() -> argparse.Namespace:
    """Optional CLI args, used only to PREFILL the sidebar inputs.

    Everything is optional now — the GUI can be launched bare and configured in
    the sidebar. ``--k_min``/``--k_max`` are accepted for backwards-compat but
    ignored (the K range always uses the defaults).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--scdata", type=str, default="")
    parser.add_argument("--stdata", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--k_step", type=int, default=_DEFAULT_K_STEP)
    parser.add_argument("--k_min", type=int, default=None)
    parser.add_argument("--k_max", type=int, default=None)
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known


@st.cache_data(show_spinner=False)
def _load_soft(root_str: str, k: int):
    return data_access.load_soft(Path(root_str), k)


@st.cache_data(show_spinner=False)
def _coords(st_path_str: str):
    return data_access.load_spatial_coords(Path(st_path_str))


@st.cache_resource(show_spinner=False)
def _scaffold_sc(sc_path_str: str, st_path_str: str, out_str: str, resolution: float):
    return scaffold.load_or_build_sc(
        Path(sc_path_str), Path(st_path_str), Path(out_str), resolution
    )


def _resolution(output_dir: Path) -> float:
    return (
        data_access.leiden_resolution_from_config(output_dir)
        or compute.DEFAULT_LEIDEN_RESOLUTION
    )


def _load_scaffold(
    args: argparse.Namespace, *, warn: str | None = None
) -> "AnnData | None":
    """Return the cached reference scaffold, or ``None`` if it fails to build.

    On failure, surface ``warn`` (with the exception appended) as an
    ``st.warning`` when given, else fail silently — callers that can render
    without the scaffold simply skip it.
    """
    try:
        return _scaffold_sc(
            str(args.scdata),
            str(args.stdata),
            str(args.output_dir),
            _resolution(args.output_dir),
        )
    except Exception as exc:  # noqa: BLE001
        if warn:
            st.warning(f"{warn}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Figure export (per-figure PNG / SVG / PDF via kaleido)
# --------------------------------------------------------------------------- #
_EXPORT_MIME = {
    "png": "image/png",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
}


def _export_popover(fig, *, key: str, stem: str) -> None:
    """A small ⬇ popover under a figure: pick a format, render it server-side
    (kaleido), and download. Kept on-demand — the image is only built when the
    user clicks *Generate*, so the (heavy) kaleido call never runs on a plain
    rerun."""
    with st.popover("⬇ Export"):
        fmt = st.radio(
            "Format",
            list(_EXPORT_MIME),
            horizontal=True,
            key=f"{key}_fmt",
            help="PNG is always faithful; the UMAP/spatial scatter panels "
            "rasterise inside SVG/PDF (WebGL), while bar/box/line/heatmap "
            "figures stay true vector.",
        )
        if st.button("Generate", key=f"{key}_gen", width="stretch"):
            st.session_state.pop(f"{key}_err", None)
            try:
                st.session_state[f"{key}_bytes"] = render.figure_to_bytes(fig, fmt)
                st.session_state[f"{key}_ext"] = fmt
            except Exception as exc:  # noqa: BLE001 - surfaced below
                st.session_state.pop(f"{key}_bytes", None)
                st.session_state[f"{key}_err"] = str(exc)

        data = st.session_state.get(f"{key}_bytes")
        if data is not None:
            ext = st.session_state.get(f"{key}_ext", "png")
            st.download_button(
                f"Download .{ext}",
                data,
                file_name=f"{stem}.{ext}",
                mime=_EXPORT_MIME.get(ext, "application/octet-stream"),
                key=f"{key}_dl",
                width="stretch",
            )
        if st.session_state.get(f"{key}_err"):
            st.error(st.session_state[f"{key}_err"])


def _plot_card(fig, *, key: str, stem: str, caption: str | None = None) -> None:
    """Render a Plotly figure as a report card: the chart, an optional caption,
    and an export popover beneath it."""
    st.plotly_chart(fig, width="stretch", key=key)
    if caption:
        st.caption(caption)
    _export_popover(fig, key=f"{key}_exp", stem=stem)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _render_card_grid(cards: list[tuple[str, Callable[[], None]]]) -> None:
    """Lay out ``(title, body)`` cards two per row, each in a bordered container."""
    for start in range(0, len(cards), 2):
        cols = st.columns(2)
        for col, (title, body) in zip(cols, cards[start : start + 2]):
            with col, st.container(border=True):
                st.markdown(f"**{title}**")
                body()


def _render_metrics(container, d: dict, level: int = 0) -> None:
    """Show a metrics dict: scalars as a two-column table, nested dicts recursively."""
    scalars = {k: v for k, v in d.items() if not isinstance(v, (dict, list))}
    if scalars:
        df = pd.DataFrame(
            {"metric": list(scalars.keys()), "value": list(scalars.values())}
        )
        container.dataframe(df, hide_index=True, width="stretch")
    for k, v in d.items():
        if isinstance(v, dict):
            container.markdown(f"{'#' * min(6, 4 + level)} {k}")
            _render_metrics(container, v, level + 1)
        elif isinstance(v, list):
            try:
                container.dataframe(pd.DataFrame(v), width="stretch")
            except Exception:  # noqa: BLE001
                container.write({k: v})


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _headline(
    mapper: str,
    root: Path,
    k: int,
    args: argparse.Namespace,
    *,
    key_prefix: str,
    show_shared_umap: bool = False,
    conf_controls: bool = False,
    threshold: float = 0.0,
    plot_confidence: bool = False,
) -> None:
    """UMAP(s) + live spatial plot sharing one state legend.

    When ``conf_controls`` is set and the mapper wrote per-spot confidence, the
    confidence-threshold slider and the "Plot confidence" checkbox are rendered
    directly below the plots, aligned under the (rightmost) spatial panel, and
    their values drive this tab's spatial colouring. Otherwise ``threshold`` /
    ``plot_confidence`` are taken from the arguments — the Compare tab passes
    neither and sets ``conf_controls=False``, so it never shows the controls.
    """
    _P, hard, confidence = _load_soft(str(root), k)
    coords = _coords(str(args.stdata))
    have_conf = confidence is not None
    have_spatial = coords is not None

    # Build the reference scaffold for the UMAP panel.
    adata_sc = _load_scaffold(args, warn="Scaffold build failed — UMAP unavailable")

    have_umap = adata_sc is not None
    # Mirror render.render_headline_figure's panel logic so the confidence
    # controls line up under the correct (rightmost) column.
    have_shared = (
        have_umap and show_shared_umap and OBSM_UMAP_SHARED_GENES in adata_sc.obsm
    )
    n_panels = int(have_shared) + int(have_umap) + int(have_spatial)

    # Reserve the plot's slot; the confidence controls render *below* it so the
    # figure stays on top and the controls sit under the spatial panel.
    plot_slot = st.container()

    if conf_controls and have_conf and have_spatial and n_panels:
        control_cols = st.columns(n_panels)
        with control_cols[-1]:
            threshold = widgets.live_slider(
                "Confidence threshold",
                0.0,
                1.0,
                0.01,
                0.0,
                key=f"{key_prefix}_thr",
            )
            plot_confidence = st.checkbox(
                "Plot confidence",
                value=False,
                key=f"{key_prefix}_pconf",
                help="Colour spots by confidence intensity instead of assigned state.",
            )

    # One figure, all subplots (UMAP(s) + spatial) sharing a single state legend.
    # Rendered by a client-side component so clicking a legend entry OR any point
    # toggles that state (single click) / isolates it (double click); inactive
    # states are drawn in a light colour rather than hidden.
    if coords is None and adata_sc is None:
        with plot_slot:
            st.info("Nothing to plot: no spatial coords and no UMAP scaffold.")
        return

    try:
        fig = render.render_headline_figure(
            coords,
            hard,
            confidence,
            threshold,
            k,
            adata_sc=adata_sc,
            root=root,
            plot_confidence=plot_confidence,
            show_shared_umap=show_shared_umap,
        )
    except Exception as exc:  # noqa: BLE001
        with plot_slot:
            st.warning(f"UMAP rendering failed — showing spatial only: {exc}")
        fig = render.render_headline_figure(
            coords,
            hard,
            confidence,
            threshold,
            k,
            adata_sc=None,
            root=root,
            plot_confidence=plot_confidence,
        )
    with plot_slot:
        widgets.headline_plot(fig, key=f"{key_prefix}_headline")
        _export_popover(
            fig, key=f"{key_prefix}_headline_exp", stem=f"{key_prefix}_k{k:03d}"
        )

    if coords is None:
        st.caption("ST data has no obsm['spatial'] — no spatial plot.")


def _render_progress(mapper: str, run: "compute.MapperRun") -> None:
    """Progress bar for a mapper currently being computed."""
    done, expected = run.n_done(), run.n_expected()
    if expected:
        st.progress(
            min(done / expected, 1.0), text=f"Computing {mapper}: {done}/{expected} K"
        )
    else:
        st.progress(0.0, text=f"Computing {mapper}: over-clustering…")
    st.caption("This tab will show results when the sweep finishes.")


class _Controls(NamedTuple):
    """Shared display controls that drive every result tab at once.

    Only K and the shared-gene-UMAP toggle are global; the confidence threshold
    and "Plot confidence" checkbox are per-tab (rendered inside each mapper tab,
    where they can be hidden when that mapper has no confidence).
    """

    k: int
    show_shared_umap: bool


def _shared_controls(ks: list[int]) -> _Controls:
    """Render the K slider and the shared-gene-UMAP toggle once, above the tabs."""
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    with c1:
        k = widgets.live_select_slider(
            "K (number of cell states)", ks, ks[0], key="ctrl_k"
        )
    with c2:
        show_shared_umap = st.checkbox(
            "Show shared-gene-only UMAP",
            value=False,
            key="ctrl_shared_umap",
            help="Add the shared-gene (ST-overlap) UMAP as a third panel, left of "
            "the all-gene UMAP.",
        )
    return _Controls(int(k), bool(show_shared_umap))


def _set_ctrl_k(idx: int) -> None:
    """on_click: move the shared K slider to option index ``idx`` (runs before the
    slider re-instantiates, so writing its session-state key is allowed)."""
    st.session_state["ctrl_k"] = {"value": int(idx)}


def _best_k(df, column: str) -> int | None:
    """The K maximising ``column`` in the sweep table (ignoring NaN), or None."""
    if column not in df.columns:
        return None
    sub = df[["k", column]].dropna()
    if sub.empty:
        return None
    return int(sub.loc[sub[column].idxmax(), "k"])


def _ksweep_section(mapper: str, root: Path, ks_all: list[int]) -> None:
    """K-sweep row (3 interactive plots) plus buttons that jump the shared K
    slider to the K optimising each criterion."""
    df = data_access.ksweep_table(root)
    if df is None or df.empty:
        return

    with st.expander("Comparing K", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            _plot_card(
                render.render_ksweep_reconstruction_figure(df),
                key=f"ks_recon_{mapper}",
                stem=f"{mapper}_ksweep_reconstruction",
            )
        with c2:
            _plot_card(
                render.render_ksweep_spatial_figure(df),
                key=f"ks_spatial_{mapper}",
                stem=f"{mapper}_ksweep_spatial",
            )
        with c3:
            _plot_card(
                render.render_ksweep_coherence_figure(df),
                key=f"ks_coherence_{mapper}",
                stem=f"{mapper}_ksweep_coherence",
            )

        # Suggested "best" K per criterion; clicking sets the shared K slider.
        suggestions = [
            ("Spot cossim (raw, hard)", _best_k(df, "cossim_hard_raw_spot")),
            ("Spatial organisation", _best_k(df, "nhood_mean_self_zscore")),
            ("Transcriptional coherence", _best_k(df, "modularity_st_expression")),
        ]
        st.caption("Jump the K slider to the K that optimises:")
        cols = st.columns(len(suggestions))
        for col, (label, bk) in zip(cols, suggestions):
            with col:
                usable = bk is not None and bk in ks_all
                st.button(
                    f"{label}  →  K = {bk}" if bk is not None else f"{label}  →  n/a",
                    key=f"bestk_{mapper}_{label}",
                    on_click=_set_ctrl_k if usable else None,
                    args=(ks_all.index(bk),) if usable else None,
                    disabled=not usable,
                    width="stretch",
                )


def _mapper_tab(
    mapper: str,
    args: argparse.Namespace,
    runs: dict,
    queue: list,
    ctrl: _Controls | None,
    ks_all: list[int],
) -> None:
    root = data_access.run_root(args.output_dir, mapper)
    run = runs.get(mapper)

    # In-progress / queued / failed states take precedence over any partial
    # on-disk output.
    if run is not None and run.is_running():
        _render_progress(mapper, run)
        return
    if mapper in queue:
        st.info(
            f"🕒 Waiting for another method to finish before computing **{mapper}**…"
        )
        return
    if run is not None and run.error and not data_access.list_ks(root):
        st.error(f"'{mapper}' failed to compute — see the terminal for details.")
        return

    ks = data_access.list_ks(root)
    if not ks:
        st.info("No K folders found yet for this mapper.")
        return
    if ctrl is None or ctrl.k not in ks:
        st.info("K not available for this method at the current shared setting.")
        return

    _ksweep_section(mapper, root, ks_all)

    st.divider()
    _headline(
        mapper,
        root,
        ctrl.k,
        args,
        key_prefix=f"tab_{mapper}",
        show_shared_umap=ctrl.show_shared_umap,
        conf_controls=True,
    )

    st.divider()
    st.subheader("Report")
    _report_dashboard(mapper, root, ctrl.k, args)


def _report_dashboard(
    mapper: str, root: Path, k: int, args: argparse.Namespace
) -> None:
    """The per-mapper report as a grid of compact cards, each an interactive
    Plotly figure or a small metrics table (no matplotlib)."""
    P, hard, confidence = _load_soft(str(root), k)

    # The fractions card is simply skipped if the scaffold can't be built.
    adata_sc = _load_scaffold(args)

    def _sharpness() -> None:
        summ = data_access.load_data_json(root, k, "onehot_summary_mapping.json")
        caption = None
        if summ and summ.get("summary"):
            s = summ["summary"]
            caption = (
                f"Gini mean {s['gini_impurity']['mean']:.3f}  ·  "
                f"entropy mean {s['entropy']['mean']:.3f}"
            )
        _plot_card(
            render.render_onehot_figure(P.max(axis=1)),
            key=f"card_sharp_{mapper}",
            stem=f"{mapper}_k{k:03d}_sharpness",
            caption=caption,
        )

    def _spatial_org() -> None:
        metrics = data_access.load_data_json(root, k, "topology_metrics.json")
        if not metrics:
            st.info("topology_metrics.json not found for this K.")
            return
        # Drop the top-level scalar table; keep the metric groups (local purity,
        # neighbourhood enrichment).
        nested = {kk: vv for kk, vv in metrics.items() if isinstance(vv, dict)}
        if nested:
            _render_metrics(st, nested)
        else:
            st.info("No spatial-organisation sub-metrics for this K.")

    def _modularity() -> None:
        metrics = data_access.load_data_json(root, k, "modularity_metrics.json")
        # Only the mapping-dependent modularity belongs here; the reference-graph
        # modularities live on the Single-cell reference tab.
        if metrics and "modularity_st_expression" in metrics:
            _render_metrics(
                st, {"modularity_st_expression": metrics["modularity_st_expression"]}
            )
        else:
            st.info("modularity_metrics.json not found for this K.")

    # Assemble the cards (title, body). Only include those that have data.
    cards: list[tuple[str, Callable[[], None]]] = []
    if adata_sc is not None:
        cards.append(
            (
                "Cell- & Spot-State Fractions",
                lambda: _plot_card(
                    render.render_fractions_figure(adata_sc, root, k, hard),
                    key=f"card_frac_{mapper}",
                    stem=f"{mapper}_k{k:03d}_fractions",
                ),
            )
        )
    cossim = data_access.load_cossim_distributions(root, k)
    if cossim:
        cards.append(
            (
                "Reconstruction Cosine Similarity",
                lambda: _plot_card(
                    render.render_reconstruction_figure(cossim),
                    key=f"card_recon_{mapper}",
                    stem=f"{mapper}_k{k:03d}_reconstruction",
                ),
            )
        )
    cards.append(("Mapping Sharpness — How One-Hot", _sharpness))
    if confidence is not None:
        cards.append(
            (
                "Mapping Confidence — Per-Spot",
                lambda: _plot_card(
                    render.render_confidence_figure(confidence),
                    key=f"card_conf_{mapper}",
                    stem=f"{mapper}_k{k:03d}_confidence",
                ),
            )
        )
    cards.append(("Spatial Organisation of Mapped Spots", _spatial_org))
    cards.append(("Modularity", _modularity))

    _render_card_grid(cards)


def _compare_tab(mappers: list[str], args: argparse.Namespace, ctrl: _Controls) -> None:
    """Compare several methods at once: one shared reference UMAP followed by each
    selected method's spatial map side by side.

    The shared-gene UMAP toggle and confidence options do not apply here (this tab
    always shows the single all-gene reference UMAP with no confidence colouring).
    """
    st.subheader("Side-by-side comparison")

    selected = st.pills(
        "Methods to compare",
        mappers,
        selection_mode="multi",
        default=list(mappers),
        key="cmp_methods",
        help="Pick two or more computed methods; their spatial maps are shown "
        "side by side under one reference UMAP.",
    )
    if len(selected) < 2:
        st.info("Select at least two methods to compare.")
        return

    # Keep the stable MAPPING_CHOICES order that `mappers` already carries.
    selected = [m for m in mappers if m in selected]
    k = ctrl.k

    coords = _coords(str(args.stdata))
    if coords is None:
        st.info("ST data has no obsm['spatial'] — nothing to compare spatially.")
        return

    # Fall back to spatial-only if the reference UMAP can't be built.
    adata_sc = _load_scaffold(
        args, warn="Scaffold build failed — reference UMAP unavailable"
    )

    # Load each selected method's hard assignment for this K (skip any missing K).
    hards: list[np.ndarray] = []
    valid: list[str] = []
    for m in selected:
        root = data_access.run_root(args.output_dir, m)
        if k not in data_access.list_ks(root):
            continue
        _P, hard, _conf = _load_soft(str(root), k)
        hards.append(hard)
        valid.append(m)
    if len(valid) < 2:
        st.info(f"Fewer than two selected methods have K={k} available.")
        return

    # One figure: reference UMAP centred on top, spatial maps in a 2-wide grid
    # below, all sharing a single interactive state legend.
    ref_root = data_access.run_root(args.output_dir, valid[0]) if adata_sc else None
    fig = render.render_compare_figure(
        coords, hards, valid, k, adata_sc=adata_sc, root=ref_root
    )
    widgets.headline_plot(fig, key="cmp_headline")
    _export_popover(fig, key="cmp_headline_exp", stem=f"compare_spatial_k{k:03d}")

    # Report sections below the plots — the same UI cards as a single method tab,
    # but with every selected method combined into one plot/table per section.
    st.divider()
    st.subheader("Report")
    _compare_sections(valid, hards, args, k)


def _compare_sections(
    valid: list[str], hards: list[np.ndarray], args: argparse.Namespace, k: int
) -> None:
    """Combined report cards for the Compare tab: each section overlays all
    selected methods in a single Plotly figure or a table with one column per
    method."""
    roots = {m: data_access.run_root(args.output_dir, m) for m in valid}

    # Gather each method's per-K data once.
    cossims: dict[str, dict] = {}
    maxprobs: dict[str, np.ndarray] = {}
    spot_fracs: dict[str, list[float]] = {}
    for m, hard in zip(valid, hards):
        P, _hard, _conf = _load_soft(str(roots[m]), k)
        cossims[m] = data_access.load_cossim_distributions(roots[m], k)
        maxprobs[m] = P.max(axis=1)
        spot_fracs[m] = [float(np.mean(hard == s)) for s in range(k)]

    def _reconstruction() -> None:
        fig = render.render_compare_reconstruction_figure(cossims)
        if fig is None:
            st.info("No reconstruction cosine-similarity data for these methods.")
        else:
            _plot_card(
                fig, key="cmp_sec_recon", stem=f"compare_reconstruction_k{k:03d}"
            )

    def _sharpness() -> None:
        _plot_card(
            render.render_compare_box_figure(
                maxprobs,
                title="Per-spot max probability (1.0 = one-hot)",
                ytitle="max probability",
            ),
            key="cmp_sec_sharp",
            stem=f"compare_sharpness_k{k:03d}",
        )

    def _fractions() -> None:
        _plot_card(
            render.render_compare_fractions_figure(spot_fracs, k),
            key="cmp_sec_frac",
            stem=f"compare_fractions_k{k:03d}",
            caption="Cell fractions are identical across methods (see a method tab).",
        )

    def _spatial_org() -> None:
        # Nested metric groups (drop the top-level scalars), methods as columns.
        cols: dict[str, dict] = {}
        for m in valid:
            topo = (
                data_access.load_data_json(roots[m], k, "topology_metrics.json") or {}
            )
            flat: dict[str, float] = {}
            for grp, sub in topo.items():
                if isinstance(sub, dict):
                    for name, val in sub.items():
                        flat[f"{grp}.{name}"] = val
            cols[m] = flat
        df = pd.DataFrame(cols)
        if df.empty:
            st.info("No spatial-organisation metrics for these methods.")
        else:
            st.dataframe(df.round(4), width="stretch")

    def _modularity() -> None:
        # Only the mapping-dependent modularity, methods as columns.
        row: dict[str, float] = {}
        for m in valid:
            mod = (
                data_access.load_data_json(roots[m], k, "modularity_metrics.json") or {}
            )
            row[m] = mod.get("modularity_st_expression", float("nan"))
        df = pd.DataFrame({"modularity_st_expression": row}).T
        st.dataframe(df.round(4), width="stretch")

    cards: list[tuple[str, "Callable[[], None]"]] = [
        ("Reconstruction Cosine Similarity", _reconstruction),
        ("Cell- & Spot-State Fractions", _fractions),
        ("Mapping Sharpness — How One-Hot", _sharpness),
    ]
    cards.append(("Spatial Organisation of Mapped Spots", _spatial_org))
    cards.append(("Modularity", _modularity))

    _render_card_grid(cards)


def _reference_tab(
    mappers: list[str], args: argparse.Namespace, ctrl: _Controls
) -> None:
    """The single-cell reference view: clustering-side plots that don't depend on
    the spot-mapping method.

    The tree cut (``leiden_to_state``) and the reference expression are identical
    across mappers, so this reads them from any finished mapper that has the
    currently selected K. Shows three UMAPs on top (Leiden overclustering, and the
    computed states on the all-gene and shared-gene embeddings), then the
    mapper-independent sections rendered as interactive Plotly.
    """
    st.subheader("Single-cell reference")

    # Any finished mapper carrying this K works — pick the first.
    ref_mapper = next(
        (
            m
            for m in mappers
            if ctrl.k in data_access.list_ks(data_access.run_root(args.output_dir, m))
        ),
        None,
    )
    if ref_mapper is None:
        st.info("No finished method has this K available yet.")
        return
    ref_root = data_access.run_root(args.output_dir, ref_mapper)

    adata_sc = _load_scaffold(
        args, warn="Scaffold build failed — reference view unavailable"
    )
    if adata_sc is None:
        return

    fig_umaps = render.render_reference_umaps_figure(adata_sc, ref_root, ctrl.k)
    widgets.headline_plot(fig_umaps, key="ref_umaps")
    _export_popover(
        fig_umaps, key="ref_umaps_exp", stem=f"reference_umaps_k{ctrl.k:03d}"
    )
    st.caption(
        f"K = {ctrl.k}. The over-clustering and its tree cut are shared across "
        f"methods (shown from **{ref_mapper}**)."
    )

    st.divider()
    with st.expander("Leiden Overclusters Merged per AIM State", expanded=True):
        _plot_card(
            render.render_leiden_merge_figure(adata_sc, ref_root, ctrl.k),
            key="ref_leiden_merge",
            stem=f"reference_leiden_merge_k{ctrl.k:03d}",
        )
    with st.expander("AIM-State Profiles", expanded=True):
        fig_prof = render.render_state_profiles_figure(adata_sc, ref_root, ctrl.k)
        if fig_prof is None:
            st.info("Too few shared genes to plot the state-profile heatmap.")
        else:
            _plot_card(
                fig_prof,
                key="ref_profiles",
                stem=f"reference_state_profiles_k{ctrl.k:03d}",
            )
    with st.expander("Substate Merge Coherence", expanded=False):
        metrics = data_access.load_data_json(ref_root, ctrl.k, "biology_metrics.json")
        if not metrics:
            st.info("biology_metrics.json not found for this K.")
        else:
            # One table for all states: state per row, metric fields as columns
            # (states that were skipped simply have NaN in the metric columns).
            per_state = metrics.get("per_state", {})
            if per_state:
                df = pd.DataFrame.from_dict(per_state, orient="index")
                df.insert(0, "state", [int(s) for s in per_state.keys()])
                df = df.sort_values("state").reset_index(drop=True).round(4)
                st.dataframe(df, hide_index=True, width="stretch")
            agg = metrics.get("aggregate") or {}
            parts = [f"n_perm = {metrics.get('n_perm')}"]
            for name, val in agg.items():
                parts.append(
                    f"{name} = {val:.4g}"
                    if isinstance(val, float)
                    else f"{name} = {val}"
                )
            st.caption("Aggregate:   " + "   •   ".join(parts))
    with st.expander("Modularity", expanded=False):
        # Reference-graph modularities only (mapper-independent); the
        # mapping-dependent modularity_st_expression lives on the method tabs.
        mod = data_access.load_data_json(ref_root, ctrl.k, "modularity_metrics.json")
        ref_keys = ["modularity_all", "modularity_shared"]
        if mod and any(kk in mod for kk in ref_keys):
            _render_metrics(st, {kk: mod[kk] for kk in ref_keys if kk in mod})
        else:
            st.info("modularity_metrics.json not found for this K.")


# --------------------------------------------------------------------------- #
# Sidebar (mapper selection + running)
# --------------------------------------------------------------------------- #
def _browse(kind: str, key: str) -> None:
    """on_click callback: open a native file/folder dialog and store the chosen
    path under session-state ``key``.

    Works because the Streamlit server runs on the user's own machine, so the
    tkinter dialog appears on their screen. Any failure (e.g. no display on a
    remote host) is surfaced as a sidebar warning; typing the path still works.
    """
    st.session_state.pop("_browse_error", None)
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            if kind == "dir":
                chosen = filedialog.askdirectory(title="Select output directory")
            else:
                chosen = filedialog.askopenfilename(
                    title="Select .h5ad file",
                    filetypes=[("AnnData h5ad", "*.h5ad"), ("All files", "*.*")],
                )
        finally:
            root.update()
            root.destroy()
    except Exception as exc:  # noqa: BLE001
        st.session_state["_browse_error"] = f"File dialog unavailable: {exc}"
        return
    if chosen:
        st.session_state[key] = chosen


def _clear_session() -> None:
    """Reset paths, K step, method selection and the run queue for a clean restart.

    Runs as a button ``on_click`` callback (before widgets re-instantiate), which
    is the only point Streamlit allows resetting widget-backed keys. On-disk
    results are left untouched; any background sweep keeps running but is no
    longer tracked.
    """
    st.session_state["cfg_sc"] = ""
    st.session_state["cfg_st"] = ""
    st.session_state["cfg_out"] = ""
    st.session_state["cfg_kstep"] = _DEFAULT_K_STEP
    st.session_state["runs"] = {}
    st.session_state["queue"] = []
    st.session_state.pop("_run_requested", None)
    # Drop transient widget state: the method pills, shared controls, and every
    # per-tab / best-K control.
    for k in list(st.session_state.keys()):
        if k == "add_methods" or k.startswith(
            ("tab_", "cmp", "ctrl_", "bestk_", "ks_")
        ):
            del st.session_state[k]


def _sidebar() -> argparse.Namespace | None:
    """Render the sidebar (data paths, K step, method selection + run queue).

    Returns the run settings once the paths are valid, else ``None``.
    """
    d = _cli_defaults()
    st.sidebar.title("AIM GUI")

    # -- data paths + K step ---------------------------------------------
    st.sidebar.subheader("Data")
    # Centre the icon inside the (stretched) Browse buttons; Streamlit left-aligns
    # button labels by default.
    st.sidebar.markdown(
        "<style>"
        "div[class*='st-key-browse_'] button{"
        "display:flex;justify-content:center;align-items:center;padding:0;}"
        "div[class*='st-key-browse_'] button>div{"
        "display:flex;justify-content:center;align-items:center;width:100%;}"
        "div[class*='st-key-browse_'] button p{margin:0;text-align:center;}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("cfg_sc", d.scdata)
    st.session_state.setdefault("cfg_st", d.stdata)
    st.session_state.setdefault("cfg_out", d.output_dir)
    st.session_state.setdefault("cfg_kstep", int(d.k_step))

    def _path_row(label: str, key: str, kind: str, browse_key: str) -> str:
        # Text field with an icon-only Browse button to its right (bottom-aligned
        # so it lines up with the input, not the label).
        col_in, col_btn = st.sidebar.columns([5, 1], vertical_alignment="bottom")
        val = col_in.text_input(label, key=key)
        col_btn.button(
            "📁",
            key=browse_key,
            on_click=_browse,
            args=(kind, key),
            help="Browse…",
            width="stretch",
        )
        return val

    sc_str = _path_row("scRNA .h5ad path", "cfg_sc", "file", "browse_sc")
    st_str = _path_row("ST .h5ad path", "cfg_st", "file", "browse_st")
    out_str = _path_row("Output directory", "cfg_out", "dir", "browse_out")
    if st.session_state.get("_browse_error"):
        st.sidebar.warning(st.session_state.pop("_browse_error"))
    k_step = st.sidebar.number_input(
        "K step",
        min_value=1,
        step=1,
        key="cfg_kstep",
        help="K min/max always use the defaults (full sweep); only the step is set here.",
    )
    st.sidebar.button(
        "Clear",
        on_click=_clear_session,
        width="stretch",
        help="Reset all inputs, the method selection and the run queue. "
        "Computed results on disk are kept.",
    )

    sc_path = Path(sc_str.strip()) if sc_str.strip() else None
    st_path = Path(st_str.strip()) if st_str.strip() else None
    out_path = Path(out_str.strip()) if out_str.strip() else None

    problems = []
    if sc_path is None or not sc_path.is_file():
        problems.append("Set a valid scRNA .h5ad path.")
    if st_path is None or not st_path.is_file():
        problems.append("Set a valid ST .h5ad path.")
    if out_path is None:
        problems.append("Set an output directory.")
    if problems:
        for p in problems:
            st.sidebar.warning(p)
        return None
    out_path.mkdir(parents=True, exist_ok=True)

    settings = argparse.Namespace(
        scdata=sc_path,
        stdata=st_path,
        output_dir=out_path,
        k_min=_DEFAULT_K_MIN,
        k_max=_DEFAULT_K_MAX,
        k_step=int(k_step),
    )

    # -- methods: locked (computed) + status + selectable remainder ------
    runs: dict[str, compute.MapperRun] = st.session_state.setdefault("runs", {})
    queue: list[str] = st.session_state.setdefault("queue", [])
    computed = data_access.list_mappers(out_path)
    running_mapper = next((m for m, r in runs.items() if r.is_running()), None)

    st.sidebar.divider()
    st.sidebar.subheader("Methods")

    # A running mapper already has config.yaml on disk (so it is in `computed`);
    # don't mark it done until its thread finishes.
    finished = [m for m in computed if m != running_mapper]
    status_lines = [f"{m}" for m in finished]
    if running_mapper:
        status_lines.append(f"⏳ {running_mapper} (running)")
    status_lines += [f"🕒 {m} (queued)" for m in queue]
    if status_lines:
        st.sidebar.markdown("\n".join(f"- {line}" for line in status_lines))

    taken = set(computed) | set(queue)
    if running_mapper:
        taken.add(running_mapper)
    remaining = [m for m in MAPPING_CHOICES if m not in taken]

    if remaining:
        selected = st.sidebar.pills(
            "Add methods to run",
            remaining,
            selection_mode="multi",
            key="add_methods",
            help="Pick one or more; they compute one after another.",
        )
        if st.sidebar.button("Run selected", disabled=not selected, width="stretch"):
            for m in selected:
                if m not in queue:
                    queue.append(m)
            # Flag a pending run so the empty-state prompt never flashes before
            # the first tab appears; cleared once any tab renders. No st.rerun()
            # here: letting main() continue in this same run starts the method and
            # renders its progress tab immediately, so the idle "select methods"
            # prompt is replaced in place instead of lingering until the next run.
            st.session_state["_run_requested"] = True
    else:
        st.sidebar.caption("All methods are computed or queued.")

    # Surface failures.
    for m, r in list(runs.items()):
        if r.error and not r.is_running() and m not in computed:
            st.sidebar.error(f"'{m}' failed — see terminal.")

    return settings


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _render_body(
    settings: argparse.Namespace,
    runs: dict,
    queue: list,
    computed: list[str],
    running_mapper: str | None,
    active: bool,
    tab_mappers: list[str],
    comparable: list[str],
) -> None:
    """Render the main content area (idle prompt, computing notice, or the tab
    set). Called inside a single ``st.empty()`` placeholder so it is replaced
    atomically each run."""
    if not tab_mappers:
        # Render EXACTLY ONE element here. The polling loop reruns via st.rerun(),
        # which does not clear stale elements, so if the idle branch emitted more
        # elements than the tabs branch, the extra ones (e.g. this prompt) would
        # linger below the tabs for the whole computation. A single st.info is a
        # different element type than st.tabs, so it is fully replaced when the
        # first tab appears.
        if active or st.session_state.get("_run_requested"):
            st.info(
                "### AIM results explorer\nComputing… tabs will appear as methods start."
            )
        else:
            st.info(
                "### AIM results explorer\n"
                "Select one or more methods in the sidebar and click **Run** to begin."
            )
        return

    # Tabs exist now — the pending-run flag has done its job.
    st.session_state.pop("_run_requested", None)
    # Shared display controls (K + shared-gene-UMAP toggle) live above the tab
    # bar and drive every tab at once. K options are the finished methods' common
    # set (all sweeps share the reference, so it matches).
    finished = [m for m in tab_mappers if m in computed and m != running_mapper]
    ks_all = sorted(
        {
            kk
            for m in finished
            for kk in data_access.list_ks(data_access.run_root(settings.output_dir, m))
        }
    )
    ctrl = _shared_controls(ks_all) if ks_all else None

    # Tab order: the "Single-cell reference" tab (clustering-side,
    # mapper-independent) first, then a divider, then the per-method tabs, then
    # the Compare tab (>=2 finished methods). The reference tab appears once any
    # method has finished.
    ref_available = ctrl is not None and len(finished) >= 1
    compare_available = len(comparable) >= 2 and ctrl is not None

    if ref_available:
        # st.tabs has no divider, so draw a vertical rule on the right edge of the
        # first tab (the reference tab) to set it apart from the methods.
        st.markdown(
            "<style>"
            'div[data-baseweb="tab-list"] button[role="tab"]:first-of-type{'
            "border-right:2px solid rgba(140,140,140,0.35);"
            "margin-right:0.75rem;padding-right:1rem;}"
            "</style>",
            unsafe_allow_html=True,
        )

    labels: list[str] = []
    if ref_available:
        labels.append("🧬 Single-cell reference")
    labels += list(tab_mappers)
    if compare_available:
        labels.append("⇄ Compare")

    tabs = st.tabs(labels)
    idx = 0
    if ref_available:
        with tabs[idx]:
            _reference_tab(finished, settings, ctrl)
        idx += 1
    for mapper in tab_mappers:
        with tabs[idx]:
            _mapper_tab(mapper, settings, runs, queue, ctrl, ks_all)
        idx += 1
    if compare_available:
        with tabs[idx]:
            _compare_tab(comparable, settings, ctrl)


def main() -> None:
    st.set_page_config(page_title="AIM GUI", layout="wide")
    settings = _sidebar()

    if settings is None:
        st.title("AIM results explorer")
        st.info(
            "Set the scRNA path, ST path, and output directory in the sidebar to begin."
        )
        return

    runs: dict[str, compute.MapperRun] = st.session_state.setdefault("runs", {})
    queue: list[str] = st.session_state.setdefault("queue", [])

    # Sequential execution: start the next queued mapper when none is running.
    running = any(r.is_running() for r in runs.values())
    just_started = False
    if not running and queue:
        nxt = queue.pop(0)
        runs[nxt] = compute.MapperRun(
            nxt,
            settings.scdata,
            settings.stdata,
            settings.output_dir,
            settings.k_min,
            settings.k_max,
            settings.k_step,
        ).start()
        running = True
        just_started = True

    computed = data_access.list_mappers(settings.output_dir)
    running_mapper = next((m for m, r in runs.items() if r.is_running()), None)
    active = running_mapper is not None or bool(queue)
    # A tab per method that is computed, running, or queued (MAPPING_CHOICES order).
    tab_set = set(computed) | set(runs.keys()) | set(queue)
    tab_mappers = [m for m in MAPPING_CHOICES if m in tab_set]
    # Comparable = finished on disk (exclude the one still running).
    comparable = [m for m in tab_mappers if m in computed and m != running_mapper]

    # Render the whole body into one st.empty() placeholder. Because st.rerun()
    # (used by the polling loop below) does NOT clear stale elements, an st.info
    # rendered on the idle run would otherwise linger on screen for the entire
    # computation. A single placeholder is re-filled atomically each run, so the
    # idle prompt is replaced the instant the first tab renders.
    with st.empty().container():
        _render_body(
            settings,
            runs,
            queue,
            computed,
            running_mapper,
            active,
            tab_mappers,
            comparable,
        )

    # While work is pending/running, refresh so progress advances and new tabs
    # flip from "waiting" to results.
    if just_started:
        # Finish this run immediately (no sleep) so Streamlit clears the now-stale
        # idle "select methods" prompt right away, instead of leaving it on screen
        # for the polling interval below. The next run does the actual polling.
        st.rerun()
    elif queue or any(r.is_running() for r in runs.values()):
        time.sleep(1.5)
        st.rerun()


# Streamlit executes this module top-to-bottom on every rerun.
main()
