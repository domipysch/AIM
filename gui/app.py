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

# Ensure the in-repo packages under src/ are importable even if PYTHONPATH was
# not inherited (e.g. when Streamlit re-execs the script).
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from aim import MAPPING_CHOICES  # noqa: E402

from gui import compute, data_access, render, scaffold  # noqa: E402
from gui.sections import SCAFFOLD_KEYS, SECTIONS, Section  # noqa: E402

_NO_CONF_MAPPERS = {"learned", "tangram", "tacco", "dot"}


# --------------------------------------------------------------------------- #
# Args & cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource
def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scdata", type=Path, required=True)
    parser.add_argument("--stdata", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--k_min", type=int, default=None)
    parser.add_argument("--k_max", type=int, default=None)
    parser.add_argument("--k_step", type=int, default=1)
    return parser.parse_args(sys.argv[1:])


@st.cache_data(show_spinner=False)
def _load_soft(root_str: str, k: int):
    return data_access.load_soft(Path(root_str), k)


@st.cache_data(show_spinner=False)
def _coords(st_path_str: str):
    return data_access.load_spatial_coords(Path(st_path_str))


@st.cache_data(show_spinner=False)
def _ksweep_csv(root_str: str):
    return data_access.ksweep_csv(Path(root_str))


@st.cache_resource(show_spinner=False)
def _scaffold_sc(sc_path_str: str, st_path_str: str, out_str: str, resolution: float):
    return scaffold.load_or_build_sc(
        Path(sc_path_str), Path(st_path_str), Path(out_str), resolution
    )


@st.cache_resource(show_spinner=False)
def _scaffold_st(st_path_str: str):
    return scaffold.read_st(st_path_str)


def _resolution(output_dir: Path) -> float:
    return (
        data_access.leiden_resolution_from_config(output_dir)
        or compute.DEFAULT_LEIDEN_RESOLUTION
    )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _render_metrics(container, d: dict, level: int = 0) -> None:
    """Show a metrics dict: scalars as a two-column table, nested dicts recursively."""
    scalars = {k: v for k, v in d.items() if not isinstance(v, (dict, list))}
    if scalars:
        df = pd.DataFrame(
            {"metric": list(scalars.keys()), "value": list(scalars.values())}
        )
        container.dataframe(df, hide_index=True, use_container_width=True)
    for k, v in d.items():
        if isinstance(v, dict):
            container.markdown(f"{'#' * min(6, 4 + level)} {k}")
            _render_metrics(container, v, level + 1)
        elif isinstance(v, list):
            try:
                container.dataframe(pd.DataFrame(v), use_container_width=True)
            except Exception:  # noqa: BLE001
                container.write({k: v})


def _confidence_supported(mapper: str) -> bool:
    return mapper not in _NO_CONF_MAPPERS


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _headline(
    mapper: str,
    root: Path,
    k: int,
    threshold: float,
    show_scaffold: bool,
    args: argparse.Namespace,
    *,
    show_ksweep: bool,
    key_prefix: str,
) -> None:
    """UMAP (scaffold) + live spatial plot, plus the K-sweep figure."""
    if show_ksweep:
        png = data_access.ksweep_png(root)
        with st.expander("K-sweep comparison (across all K)", expanded=False):
            if png is not None:
                st.image(str(png), use_container_width=True)
            else:
                st.info("k_comparison.png not found yet.")

    _P, hard, confidence = _load_soft(str(root), k)
    coords = _coords(str(args.stdata))

    left, right = st.columns(2)

    with left:
        st.caption("Reference UMAP — computed AIM states")
        if show_scaffold:
            try:
                adata_sc = _scaffold_sc(
                    str(args.scdata),
                    str(args.stdata),
                    str(args.output_dir),
                    _resolution(args.output_dir),
                )
                adata_st = _scaffold_st(str(args.stdata))
                plots = render.ensure_scaffold_plots(
                    adata_sc, adata_st, args.output_dir, mapper, root, k
                )
                if "umap_computed_state" in plots:
                    st.image(
                        str(plots["umap_computed_state"]), use_container_width=True
                    )
                else:
                    st.info("UMAP could not be rendered for this K.")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Scaffold/UMAP rendering failed: {exc}")
        else:
            st.info("UMAP disabled (enable scaffold sections in the sidebar).")

    with right:
        st.caption("Spatial — spots below confidence threshold are grey")
        if coords is None:
            st.info("ST data has no obsm['spatial'] — no spatial plot.")
        else:
            fig = render.render_spatial_fig(coords, hard, confidence, threshold, k)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        if confidence is None:
            st.caption(
                "No per-spot confidence for this mapper — threshold has no effect."
            )


def _section_body(
    section: Section,
    mapper: str,
    root: Path,
    k: int,
    show_scaffold: bool,
    args: argparse.Namespace,
) -> None:
    if section.kind == "scaffold":
        if not show_scaffold:
            st.info("Scaffold sections disabled in the sidebar.")
            return
        try:
            adata_sc = _scaffold_sc(
                str(args.scdata),
                str(args.stdata),
                str(args.output_dir),
                _resolution(args.output_dir),
            )
            adata_st = _scaffold_st(str(args.stdata))
            plots = render.ensure_scaffold_plots(
                adata_sc, adata_st, args.output_dir, mapper, root, k
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Rendering failed: {exc}")
            return
        for name in section.plots:
            if name in plots:
                st.image(str(plots[name]), use_container_width=True)

    elif section.kind == "disk":
        plots = render.ensure_disk_plots(args.output_dir, mapper, root, k)
        shown = False
        for name in section.plots:
            if name in plots:
                st.image(str(plots[name]), use_container_width=True)
                shown = True
        if not shown:
            st.info("No figure for this section (mapper wrote no data for it).")
        if section.table:
            summary = data_access.load_data_json(root, k, section.table)
            if summary:
                _render_metrics(st, summary)

    elif section.kind == "table":
        metrics = data_access.load_data_json(root, k, section.table)
        if metrics:
            _render_metrics(st, metrics)
        else:
            st.info(f"{section.table} not found for this K.")


def _mapper_tab(mapper: str, show_scaffold: bool, args: argparse.Namespace) -> None:
    root = data_access.run_root(args.output_dir, mapper)
    ks = data_access.list_ks(root)
    if not ks:
        st.info("No K folders found yet for this mapper.")
        return

    c1, c2 = st.columns([3, 2])
    with c1:
        k = st.select_slider(
            "K (number of cell states)", options=ks, key=f"tab_{mapper}_k"
        )
    with c2:
        if _confidence_supported(mapper) and data_access.has_confidence(root, k):
            threshold = st.slider(
                "Confidence threshold", 0.0, 1.0, 0.0, 0.01, key=f"tab_{mapper}_thr"
            )
        else:
            threshold = 0.0
            st.caption("Confidence threshold unavailable for this mapper.")

    _headline(
        mapper,
        root,
        k,
        threshold,
        show_scaffold,
        args,
        show_ksweep=True,
        key_prefix=f"tab_{mapper}",
    )

    st.divider()
    st.subheader("Report sections")
    for section in SECTIONS:
        with st.expander(section.title, expanded=False):
            _section_body(section, mapper, root, k, show_scaffold, args)


def _compare_tab(
    mappers: list[str], show_scaffold: bool, args: argparse.Namespace
) -> None:
    st.subheader("Side-by-side comparison")
    csel1, csel2 = st.columns(2)
    with csel1:
        left_m = st.selectbox("Left mapper", mappers, index=0, key="cmp_left")
    with csel2:
        right_m = st.selectbox(
            "Right mapper", mappers, index=min(1, len(mappers) - 1), key="cmp_right"
        )

    roots = {m: data_access.run_root(args.output_dir, m) for m in (left_m, right_m)}
    ks_sets = [set(data_access.list_ks(roots[m])) for m in (left_m, right_m)]
    shared_ks = sorted(ks_sets[0] & ks_sets[1]) or sorted(ks_sets[0] | ks_sets[1])
    if not shared_ks:
        st.info("No K folders to compare.")
        return

    ctl1, ctl2 = st.columns([3, 2])
    with ctl1:
        k = st.select_slider(
            "Shared K (number of cell states)", options=shared_ks, key="cmp_k"
        )
    with ctl2:
        threshold = st.slider(
            "Shared confidence threshold", 0.0, 1.0, 0.0, 0.01, key="cmp_thr"
        )

    section_titles = ["(none)"] + [s.title for s in SECTIONS]
    chosen = st.selectbox(
        "Section to compare", section_titles, index=0, key="cmp_section"
    )

    cols = st.columns(2)
    for col, mapper in zip(cols, (left_m, right_m)):
        with col:
            st.markdown(f"### {mapper}")
            root = roots[mapper]
            if k not in data_access.list_ks(root):
                st.info(f"K={k} not available for {mapper}.")
                continue
            _headline(
                mapper,
                root,
                k,
                threshold,
                show_scaffold,
                args,
                show_ksweep=False,
                key_prefix=f"cmp_{mapper}",
            )
            if chosen != "(none)":
                section = next(s for s in SECTIONS if s.title == chosen)
                st.markdown(f"**{section.title}**")
                _section_body(section, mapper, root, k, show_scaffold, args)


# --------------------------------------------------------------------------- #
# Sidebar (mapper selection + running)
# --------------------------------------------------------------------------- #
def _sidebar(args: argparse.Namespace) -> bool:
    st.sidebar.title("AIM GUI")
    st.sidebar.caption("Reference / ST pair")
    st.sidebar.code(
        f"sc: {args.scdata.name}\nst: {args.stdata.name}\n"
        f"out: {args.output_dir}\nK: {args.k_min}..{args.k_max} step {args.k_step}",
    )

    if "runs" not in st.session_state:
        st.session_state.runs = {}
    runs: dict[str, compute.MapperRun] = st.session_state.runs

    ready = data_access.list_mappers(args.output_dir)
    any_active = any(r.is_running() for r in runs.values())

    st.sidebar.divider()
    st.sidebar.subheader("Run a mapper")
    mapper = st.sidebar.selectbox("Mapper method", MAPPING_CHOICES, index=0)
    if mapper in ready:
        st.sidebar.caption(f"'{mapper}' already computed — re-running overwrites it.")
    run_clicked = st.sidebar.button(
        f"Run '{mapper}' for all K",
        disabled=any_active,
        use_container_width=True,
    )
    if run_clicked and not any_active:
        runs[mapper] = compute.MapperRun(
            mapper,
            args.scdata,
            args.stdata,
            args.output_dir,
            args.k_min,
            args.k_max,
            args.k_step,
        ).start()
        st.rerun()

    # Progress for the active run.
    if any_active:
        st.sidebar.divider()
        for m, r in runs.items():
            if not r.is_running():
                continue
            done, expected = r.n_done(), r.n_expected()
            if expected:
                st.sidebar.progress(
                    min(done / expected, 1.0), text=f"{m}: {done}/{expected} K"
                )
            else:
                st.sidebar.progress(0.0, text=f"{m}: over-clustering…")
    else:
        for m, r in list(runs.items()):
            if r.error:
                st.sidebar.error(f"'{m}' failed — see terminal.")

    st.sidebar.divider()
    show_scaffold = st.sidebar.checkbox(
        "UMAP / profile sections",
        value=True,
        help="Builds the reference UMAP scaffold (slower on first use). "
        "Disable for spatial + metrics only.",
    )
    st.sidebar.caption(f"Computed: {', '.join(ready) if ready else '(none yet)'}")
    return show_scaffold


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="AIM GUI", layout="wide")
    args = _get_args()

    show_scaffold = _sidebar(args)

    ready = data_access.list_mappers(args.output_dir)
    any_active = any(r.is_running() for r in st.session_state.get("runs", {}).values())

    if not ready:
        st.title("AIM results explorer")
        if any_active:
            st.info("Computing… the first mapper will appear here when finished.")
        else:
            st.info("Select a mapper in the sidebar and click **Run** to begin.")
    else:
        tab_labels = list(ready) + (["⇄ Compare"] if len(ready) >= 2 else [])
        tabs = st.tabs(tab_labels)
        for tab, mapper in zip(tabs, ready):
            with tab:
                _mapper_tab(mapper, show_scaffold, args)
        if len(ready) >= 2:
            with tabs[-1]:
                _compare_tab(ready, show_scaffold, args)

    # While a sweep runs, refresh so the progress bar advances and new K folders appear.
    if any_active:
        time.sleep(1.5)
        st.rerun()


# Streamlit executes this module top-to-bottom on every rerun.
main()
