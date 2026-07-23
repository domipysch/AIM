"""Typst PDF report generation for the post-mapping analysis."""

from __future__ import annotations

import csv
import json
import logging
import math
import typst
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_TODAY = date.today().isoformat()


def _esc(s: str) -> str:
    """Escape characters special in Typst content blocks."""
    return (
        s.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("#", "\\#")
        .replace("@", "\\@")
        .replace("_", "\\_")
    )


def _fmt(v: str) -> str:
    """Format numeric strings: integers without decimals, floats to 4 d.p.; pass strings through _esc."""
    try:
        f = float(v)
        if f == int(f) and "." not in v:
            return str(int(f))
        return f"{f:.4f}"
    except ValueError:
        return _esc(v)


def _csv_table(csv_path: Path) -> str:
    """Return a Typst #table(...) string from a CSV file."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return "#text[_(no data)_]"

    n = len(rows[0])
    lines = [
        "#table(",
        f"  columns: {n},",
        "  stroke: 0.5pt,",
        "  fill: (_, row) => if row == 0 { luma(220) } else if calc.odd(row) { luma(248) } else { white },",
        "  align: (col, _) => if col == 0 { left } else { right },",
    ]

    for ri, row in enumerate(rows):
        cells = []
        for ci, v in enumerate(row):
            # first row → bold header; first col → escaped label; rest → numeric fmt
            text = _esc(v) if (ri == 0 or ci == 0) else _fmt(v)
            cell = f"[*{text}*]" if ri == 0 else f"[{text}]"
            cells.append(cell)
        lines.append("  " + ", ".join(cells) + ",")

    lines.append(")")
    return "\n".join(lines)


def _num(v, nd: int = 4) -> str:
    """Format a value as a fixed-decimal string; non-finite / non-numeric -> 'n/a'."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"{f:.{nd}f}" if math.isfinite(f) else "n/a"


def _pct(v) -> str:
    """Format a fraction as a percentage; non-finite / non-numeric -> 'n/a'."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"{f:.1%}" if math.isfinite(f) else "n/a"


def _multi_col_table(header: tuple[str, ...], rows: list[list[str]]) -> str:
    """Return an n-column Typst #table(...) string (bold header, first col left-aligned)."""
    n = len(header)
    lines = [
        "#table(",
        f"  columns: {n},",
        "  stroke: 0.5pt,",
        "  fill: (_, row) => if row == 0 { luma(220) } else if calc.odd(row) { luma(248) } else { white },",
        "  align: (col, _) => if col == 0 { left } else { right },",
        "  " + ", ".join(f"[*{_esc(str(h))}*]" for h in header) + ",",
    ]
    for row in rows:
        lines.append("  " + ", ".join(f"[{_esc(str(c))}]" for c in row) + ",")
    lines.append(")")
    return "\n".join(lines)


def _two_col_table(
    rows: list[tuple[str, str]], header: tuple[str, str] = ("Metric", "Value")
) -> str:
    """Return a two-column Typst #table(...) string from a list of (label, value) tuples."""
    lines = [
        "#table(",
        "  columns: 2,",
        "  stroke: 0.5pt,",
        "  fill: (_, row) => if row == 0 { luma(220) } else if calc.odd(row) { luma(248) } else { white },",
        "  align: (col, _) => if col == 0 { left } else { right },",
        f"  [*{_esc(header[0])}*], [*{_esc(header[1])}*],",
    ]
    for label, value in rows:
        lines.append(f"  [{_esc(label)}], [{_esc(str(value))}],")
    lines.append(")")
    return "\n".join(lines)


def _onehot_summary_table(json_path: Path, row_label: str) -> str:
    """Two-column Typst table from a one-hotness summary JSON."""
    with open(json_path, encoding="utf-8") as fh:
        d = json.load(fh)
    s = d["summary"]
    unit = row_label.capitalize()
    rows = [
        (f"{unit}s", str(d["n_rows"])),
        ("Max-prob mean", f"{s['max_prob']['mean']:.4f}"),
        ("Max-prob median", f"{s['max_prob']['median']:.4f}"),
        ("Gini impurity mean", f"{s['gini_impurity']['mean']:.4f}"),
        ("Entropy mean", f"{s['entropy']['mean']:.4f}"),
        (f"{unit}s with max-prob > 0.5", f"{s['frac_max_prob_above_0.5']:.1%}"),
        (f"{unit}s with max-prob > 0.9", f"{s['frac_max_prob_above_0.9']:.1%}"),
        (f"{unit}s with max-prob > 0.99", f"{s['frac_max_prob_above_0.99']:.1%}"),
    ]
    return _two_col_table(rows)


def _onehot_section(
    plots_dir: Path,
    data_dir: Path,
    base: Path,
    title: str,
    row_label: str,
) -> str:
    """One-hotness section for the spot->state mapping P from the
    onehot_*_mapping.* files. Empty string if the distribution plot is missing."""
    dist_png = plots_dir / "onehot_distribution_mapping.png"
    if not dist_png.exists():
        return ""
    thresh_png = plots_dir / "onehot_thresholds_mapping.png"
    if thresh_png.exists():
        img_block = (
            "#grid(columns: 2, column-gutter: 8pt, align: horizon,\n"
            f"  [{_img(dist_png, base)}],\n"
            f"  [{_img(thresh_png, base)}],\n"
            ")\n"
        )
    else:
        img_block = f"{_img(dist_png, base)}\n"
    sec = f"\n= {title}\n\n{img_block}"
    summary_json = data_dir / "onehot_summary_mapping.json"
    table_block = (
        "\n" + _onehot_summary_table(summary_json, row_label) + "\n"
        if summary_json.exists()
        else ""
    )
    return sec + table_block


def _img_path(p: Path, base: Path) -> str:
    """Return a Typst-safe image path relative to *base* (the .typ file's directory)."""
    try:
        rel = p.resolve().relative_to(base.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        # Fallback: file:// URI avoids drive-letter issues on Windows
        return p.resolve().as_uri()


def _img(p: Path, base: Path, width: str = "100%") -> str:
    """Return a Typst #image(...) directive for path p, relative to base."""
    return f'#image("{_img_path(p, base)}", width: {width})'


def _section_img(p: Path, title: str, base: Path, width: str = "100%") -> str:
    """Return a Typst heading + image section, or '' if the file is absent."""
    if not p.exists():
        return ""
    return f"\n= {title}\n\n{_img(p, base, width)}\n"


def _compile(source: str, out_pdf: Path) -> bool:
    """Compile a Typst source string to PDF via the typst Python bindings.
    Writes a temporary .typ file beside the PDF so relative image paths resolve correctly.
    """
    typ_file = out_pdf.with_suffix(".typ")
    typ_file.write_text(source, encoding="utf-8")
    try:
        pdf_bytes = typst.compile(str(typ_file), format="pdf")
        out_pdf.write_bytes(pdf_bytes)
        logger.info("Report → %s", out_pdf)
        return True
    except Exception as e:
        logger.error("typst compile failed: %s", e)
        return False
    finally:
        typ_file.unlink(missing_ok=True)


_PAGE_SETUP = """\
#set page(paper: "a4", margin: (top: 2cm, bottom: 2cm, left: 2.5cm, right: 2.5cm))
#set text(size: 11pt)
#set heading(numbering: "1.")
"""


def _load_metrics_json(data_dir: Path, filename: str) -> dict | None:
    """Load a metrics JSON from the analysis data dir, or None if it is absent
    or unreadable."""
    json_path = data_dir / filename
    if not json_path.exists():
        return None
    try:
        with open(json_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _spatial_organization_section(data_dir: Path, plots_dir: Path, base: Path) -> str:
    """Spatial-organisation-of-mapped-spots section from topology_metrics.json.
    Returns '' if the file or the spatial metrics are absent."""
    sp = _load_metrics_json(data_dir, "topology_metrics.json")
    if sp is None:
        return ""
    lsp = sp.get("local_purity")
    if not lsp:
        return ""

    def _spatial_row(name: str, d: dict | None) -> list[str]:
        if not d:
            return [name, "n/a", "n/a", "n/a", "n/a"]
        return [
            name,
            _num(d.get("observed")),
            _num(d.get("null_mean")),
            _num(d.get("z_score"), 2),
            _num(d.get("p_value"), 3),
        ]

    header: tuple[str, ...] = ("Metric", "Observed", "Null mean", "z-score", "p-value")
    rows = [
        _spatial_row("Local spatial purity", lsp),
    ]

    # Neighbourhood enrichment (squidpy): mean diagonal z-score summarises how much
    # each mapped state preferentially borders its own kind.
    nh = sp.get("nhood_enrichment")
    nhood_block = ""
    if nh:
        nhood_png = plots_dir / "nhood_enrichment.png"
        img = f"\n{_img(nhood_png, base, width='70%')}\n" if nhood_png.exists() else ""
        nhood_block = (
            f"\nNeighbourhood enrichment (squidpy, {nh.get('n_states', '?')} states): "
            f"mean self z-score = {_num(nh.get('mean_self_zscore'), 2)}, "
            f"mean cross z-score = {_num(nh.get('mean_cross_zscore'), 2)}. "
            "A high self value = states preferentially border their own kind.\n" + img
        )

    return (
        "\n= Spatial Organisation of Mapped Spots\n\n"
        "Mapped spot states (argmax of P) scored against a label-shuffle null "
        f"(spots: {sp.get('n_spots', '?')}, k neighbours: {sp.get('k', '?')}, "
        f"permutations: {sp.get('n_perm', '?')}). Higher purity / "
        "z-score = mapped states are more spatially coherent than chance.\n\n"
        + _multi_col_table(header, rows)
        + "\n"
        + nhood_block
    )


def _substate_coherence_section(data_dir: Path) -> str:
    """Substate-merge-coherence section from biology_metrics.json.
    Returns '' if the file or the coherence aggregate is absent."""
    coh = _load_metrics_json(data_dir, "biology_metrics.json")
    if coh is None:
        return ""
    agg = coh.get("aggregate", {})
    per_state = coh.get("per_state", {})
    if not agg:
        return ""

    agg_tbl = _two_col_table(
        [
            (
                "States tested (merge >=2 Leiden clusters)",
                str(int(agg.get("n_tested_states", 0))),
            ),
            ("Mean pairwise cosine similarity", _num(agg.get("mean_cossim"))),
            ("Mean z-score vs. null", _num(agg.get("mean_z_score"), 2)),
            ("Fraction significant (p < 0.05)", _pct(agg.get("frac_significant"))),
        ]
    )
    tested = {s: m for s, m in per_state.items() if m.get("skipped_reason") is None}
    per_state_block = ""
    if tested:
        header = (
            "State",
            "n Leiden",
            "Mean cossim",
            "Median",
            "Null mean",
            "z-score",
            "p-value",
        )
        rows = [
            [
                s,
                str(m.get("n_leiden_sub", "?")),
                _num(m.get("mean_cossim")),
                _num(m.get("median_cossim")),
                _num(m.get("null_mean")),
                _num(m.get("z_score_mean"), 2),
                _num(m.get("p_value_mean"), 3),
            ]
            for s, m in sorted(tested.items(), key=lambda kv: int(kv[0]))
        ]
        per_state_block = "\n" + _multi_col_table(header, rows) + "\n"
    n_skipped = len(per_state) - len(tested)
    skipped_note = (
        f"\n\n#text(size: 9pt, fill: luma(120))[{n_skipped} state(s) not testable "
        "(fewer than 2 merged Leiden clusters, or none in other states to form a null).]"
        if n_skipped
        else ""
    )
    return (
        "\n= Substate Merge Coherence (shared genes)\n\n"
        "For each computed state merging >=2 Leiden clusters: mean pairwise cosine "
        "similarity of the merged clusters' shared-gene centroids, vs. a null of "
        "same-sized random draws of Leiden clusters from other states. Higher cosine / "
        "z-score = the merged clusters are more mutually alike than an arbitrary "
        "same-sized group, i.e. the merge is more coherent."
        f"{skipped_note}\n\n" + agg_tbl + per_state_block
    )


def generate_analysis_report(
    analysis_dir: Path,
) -> Path | None:
    """Generate a PDF report for one analysis run and return its path."""
    analysis_dir = Path(analysis_dir)
    plots_dir = analysis_dir / "plots"
    data_dir = analysis_dir / "data"
    out_pdf = analysis_dir / f"report.pdf"

    parts: list[str] = [_PAGE_SETUP]

    parts.append(f"""
#align(center)[
  #text(size: 22pt, weight: "bold")[Analysis Report]
  #v(0.3em)
  #text(size: 9pt, fill: luma(120))[Generated {_TODAY}]
]
#v(0.8em)
#line(length: 100%)
#v(0.5em)
""")

    base = analysis_dir  # image paths in the .typ are relative to this dir

    # Spatial distribution of AIM states (computed-state UMAP beside it)
    spatial_png = plots_dir / "spatial_cell_states.png"
    umap_cs_png = plots_dir / "umap_computed_state.png"
    if spatial_png.exists() and umap_cs_png.exists():
        parts.append(
            "\n= Spatial Distribution of AIM States\n\n"
            "#grid(columns: 2, column-gutter: 8pt, align: horizon,\n"
            f"  [{_img(umap_cs_png, base)}],\n"
            f"  [{_img(spatial_png, base)}],\n"
            ")\n"
        )
    else:
        sec = _section_img(spatial_png, "Spatial Distribution of AIM States", base)
        if sec:
            parts.append(sec)

    # Cell- and spot-state fractions
    sec = _section_img(
        plots_dir / "cell_state_fractions.png",
        "Cell- and Spot-State Fractions (AIM States)",
        base,
    )
    if sec:
        parts.append(sec)

    # UMAP grid — Leiden overclusters vs computed states, all vs shared genes
    sec = _section_img(
        plots_dir / "umap_grid.png",
        "UMAP — Leiden Overclusters vs Computed States (all & shared genes)",
        base,
    )
    if sec:
        parts.append(sec)

    # Which Leiden overclusters were merged into each AIM state
    sec = _section_img(
        plots_dir / "leiden_merge_map.png",
        "Leiden Overclusters Merged per AIM State",
        base,
    )
    if sec:
        parts.append(sec)

    # Spatial organisation of the mapped spots
    parts.append(_spatial_organization_section(data_dir, plots_dir, base))

    # Reconstruction cosine similarity
    cossim_csv = data_dir / "cossim_summary.csv"
    if cossim_csv.exists():
        boxplot_png = plots_dir / "cossim_boxplots.png"
        boxplot_block = f"\n{_img(boxplot_png, base)}\n" if boxplot_png.exists() else ""
        parts.append(f"""
= Reconstruction Cosine Similarity

Predicted spot expression is reconstructed as `mapping @ state_centroids`,
where AIM-state centroids are assembled from the Leiden-cluster expression
sums, weighted by the Leiden-cluster -> AIM-state merge matrix G. "soft" uses
G and the spot mapping P as given; "hard" uses their row-wise argmax one-hot
versions (paired together, matching AIM's original deterministic-mode
semantics). "raw" compares raw counts; "norm" compares total-count-normalized
+ log1p-transformed data.

{_csv_table(cossim_csv)}
{boxplot_block}""")

    # AIM-state profiles
    sec = _section_img(
        plots_dir / "cell_state_profiles.png", "AIM-State Profiles", base
    )
    if sec:
        parts.append(sec)

    # Mapping sharpness — spot -> AIM state mapping
    parts.append(
        _onehot_section(
            plots_dir,
            data_dir,
            base,
            'Mapping Sharpness — Spot → AIM State Mapping ("How One-Hot")',
            "spot",
        )
    )

    # Substate merge coherence
    parts.append(_substate_coherence_section(data_dir))

    source = "\n".join(p for p in parts if p)
    return out_pdf if _compile(source, out_pdf) else None
