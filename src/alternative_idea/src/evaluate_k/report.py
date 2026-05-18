"""
PDF report generation via Typst for post-mapping analysis.

Public API
----------
generate_per_k_report(analysis_dir, K, run_id)  -> Path | None
generate_summary_report(pair_dir)               -> Path | None

Returns None (with a warning) when `typst` is not on PATH or compilation fails.
The .typ source file is kept alongside the PDF for debugging.
"""

from __future__ import annotations

import csv
import logging
import subprocess
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_TODAY = date.today().isoformat()


# ─── Low-level helpers ────────────────────────────────────────────────────────


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


def _two_col_table(
    rows: list[tuple[str, str]], header: tuple[str, str] = ("Metric", "Value")
) -> str:
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


def _csv_table_with_extra(
    csv_path: Path,
    extra_rows: list[tuple[str, str] | None] | None = None,
) -> str:
    """Two-column Typst table from a CSV file with optional appended rows."""
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
            text = _esc(v) if (ri == 0 or ci == 0) else _fmt(v)
            cell = f"[*{text}*]" if ri == 0 else f"[{text}]"
            cells.append(cell)
        lines.append("  " + ", ".join(cells) + ",")

    if extra_rows:
        for item in extra_rows:
            if item is None:
                continue
            label, value = item
            lines.append(f"  [{_esc(label)}], [{_esc(value)}],")

    lines.append(")")
    return "\n".join(lines)


_GREEN = 'rgb("#d4f5d4")'
_RED = 'rgb("#f5d4d4")'


def _substate_metrics_table(csv_path: Path) -> str:
    """
    Typst table for substate_metrics.csv with conditional cell colouring:
      - perm_p  : green < 0.33, red >= 0.33
      - var_act / var_null : green when var_act < var_null, red otherwise (both cells)
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return "#text[_(no data)_]"

    header = rows[0]
    n = len(header)

    def _col(name: str) -> int:
        return header.index(name) if name in header else -1

    perm_p_col = _col("perm_p")
    var_act_col = _col("var_act")
    var_null_col = _col("var_null")

    lines = [
        "#table(",
        f"  columns: {n},",
        "  stroke: 0.5pt,",
        "  fill: (_, row) => if row == 0 { luma(220) } else if calc.odd(row) { luma(248) } else { white },",
        "  align: (col, _) => if col == 0 { left } else { right },",
    ]

    # Header row
    lines.append("  " + ", ".join(f"[*{_esc(h)}*]" for h in header) + ",")

    # Data rows
    for row in rows[1:]:
        col_fill: dict[int, str] = {}

        if perm_p_col >= 0 and perm_p_col < len(row):
            try:
                col_fill[perm_p_col] = _GREEN if float(row[perm_p_col]) < 0.33 else _RED
            except ValueError:
                pass

        if (
            var_act_col >= 0
            and var_null_col >= 0
            and var_act_col < len(row)
            and var_null_col < len(row)
        ):
            try:
                fill = (
                    _GREEN
                    if float(row[var_act_col]) < float(row[var_null_col])
                    else _RED
                )
                col_fill[var_act_col] = fill
                col_fill[var_null_col] = fill
            except ValueError:
                pass

        cells = []
        for ci, v in enumerate(row):
            text = _esc(v) if ci == 0 else _fmt(v)
            if ci in col_fill:
                cells.append(f"table.cell(fill: {col_fill[ci]})[{text}]")
            else:
                cells.append(f"[{text}]")
        lines.append("  " + ", ".join(cells) + ",")

    lines.append(")")
    return "\n".join(lines)


def _mapping_clarity_table(csv_path: Path) -> str:
    """
    Typst table for mapping_clarity.csv.
    Both value cells are coloured green when margin_computed_mean > margin_leiden_mean.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return "#text[_(no data)_]"

    # Locate the two metric rows by name
    data = {r[0]: r[1] for r in rows[1:] if len(r) >= 2}
    try:
        computed_margin = float(data.get("margin_computed_mean", "nan"))
        leiden_margin = float(data.get("margin_leiden_mean", "nan"))
        value_fill = _GREEN if computed_margin > leiden_margin else ""
    except ValueError:
        value_fill = ""

    header = rows[0]
    n = len(header)

    lines = [
        "#table(",
        f"  columns: {n},",
        "  stroke: 0.5pt,",
        "  fill: (_, row) => if row == 0 { luma(220) } else if calc.odd(row) { luma(248) } else { white },",
        "  align: (col, _) => if col == 0 { left } else { right },",
        "  " + ", ".join(f"[*{_esc(h)}*]" for h in header) + ",",
    ]

    for row in rows[1:]:
        cells = []
        for ci, v in enumerate(row):
            text = _esc(v) if ci == 0 else _fmt(v)
            if ci > 0 and value_fill:
                cells.append(f"table.cell(fill: {value_fill})[{text}]")
            else:
                cells.append(f"[{text}]")
        lines.append("  " + ", ".join(cells) + ",")

    lines.append(")")
    return "\n".join(lines)


def _img_path(p: Path, base: Path) -> str:
    """Return a Typst-safe image path relative to *base* (the .typ file's directory)."""
    try:
        rel = p.resolve().relative_to(base.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        # Fallback: file:// URI avoids drive-letter issues on Windows
        return p.resolve().as_uri()


def _img(p: Path, base: Path, width: str = "100%") -> str:
    return f'#image("{_img_path(p, base)}", width: {width})'


def _section_img(p: Path, title: str, base: Path, width: str = "100%") -> str:
    """Return a Typst heading + image section, or '' if the file is absent."""
    if not p.exists():
        return ""
    return f"\n= {title}\n\n{_img(p, base, width)}\n"


def _compile(source: str, out_pdf: Path) -> bool:
    """Write source to a .typ file beside out_pdf and compile with typst."""
    typ_file = out_pdf.with_suffix(".typ")
    typ_file.write_text(source, encoding="utf-8")
    try:
        r = subprocess.run(
            ["typst", "compile", str(typ_file), str(out_pdf)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            logger.error("typst compile failed:\n%s", r.stderr)
            return False
        logger.info("Report → %s", out_pdf)
        return True
    except FileNotFoundError:
        logger.warning("typst not found on PATH — skipping PDF report generation.")
        typ_file.unlink(missing_ok=True)
        return False
    except subprocess.TimeoutExpired:
        logger.error("typst compile timed out for %s", out_pdf)
        return False


_PAGE_SETUP = """\
#set page(paper: "a4", margin: (top: 2cm, bottom: 2cm, left: 2.5cm, right: 2.5cm))
#set text(size: 11pt)
#set heading(numbering: "1.")
"""


# ─── Per-K report ─────────────────────────────────────────────────────────────


def generate_per_k_report(
    analysis_dir: Path,
    K: int,
    run_id: str,
    median_cossim_gene: float | None = None,
    median_cossim_spot: float | None = None,
) -> Path | None:
    """Generate a PDF report for one analysis run and return its path."""
    analysis_dir = Path(analysis_dir)
    out_pdf = analysis_dir / f"report_K{K}.pdf"

    parts: list[str] = [_PAGE_SETUP]

    # Title block
    parts.append(f"""
#align(center)[
  #text(size: 22pt, weight: "bold")[Analysis Report]
  #v(0.3em)
  #text(size: 15pt)[K = {K} — Run {run_id}]
  #v(0.3em)
  #text(size: 9pt, fill: luma(120))[Generated {_TODAY}]
]
#v(0.8em)
#line(length: 100%)
#v(0.5em)
""")

    # Unsupervised metrics
    unsup = analysis_dir / "unsupervised_metrics.csv"
    if unsup.exists():
        parts.append(f"""
= Unsupervised Metrics

{_csv_table(unsup)}
""")

    # Supervised metrics
    sup = analysis_dir / "supervised_metrics.csv"
    if sup.exists():
        parts.append(f"""
= Supervised Metrics

{_csv_table(sup)}
""")

    # Per-state sub-cluster analysis
    substate_csv = analysis_dir / "substate_metrics.csv"
    if substate_csv.exists():
        parts.append(f"""
= Per-State Sub-cluster Analysis

#set text(size: 7pt)
{_substate_metrics_table(substate_csv)}
#set text(size: 11pt)
""")

    # Spot-level mapping clarity
    clarity_csv = analysis_dir / "mapping_clarity.csv"
    if clarity_csv.exists():
        parts.append(f"""
= Spot-level Mapping Clarity

{_mapping_clarity_table(clarity_csv)}
""")

    # O2 cosine similarity
    if median_cossim_gene is not None or median_cossim_spot is not None:
        o2_rows: list[tuple[str, str]] = []
        if median_cossim_gene is not None:
            o2_rows.append(("Genewise median cosine sim", f"{median_cossim_gene:.4f}"))
        if median_cossim_spot is not None:
            o2_rows.append(("Spotwise median cosine sim", f"{median_cossim_spot:.4f}"))
        parts.append(f"""
= O2 Cosine Similarity

{_two_col_table(o2_rows)}
""")

    base = analysis_dir  # .typ lives here; image paths are relative to it

    # UMAP comparison
    parts.append(
        _section_img(analysis_dir / "umap_comparison.png", "UMAP Comparison", base)
    )

    # GEP centroid distance comparison
    parts.append(
        _section_img(
            analysis_dir / "gep_distance_comparison.png",
            "GEP Pairwise Cosine Distance",
            base,
        )
    )

    # Cell-state profiles (combined heatmap + bars)
    parts.append(
        _section_img(
            analysis_dir / "cell_state_profiles.png", "Cell-State Profiles", base
        )
    )

    # Fraction bar charts standalone
    parts.append(
        _section_img(
            analysis_dir / "cell_state_fractions.png",
            "Cell- and Spot-State Fractions",
            base,
        )
    )

    # Centroid matching — side-by-side when both exist
    hung = analysis_dir / "centroid_matching_hungarian.png"
    greedy = analysis_dir / "centroid_matching_greedy.png"
    if hung.exists() and greedy.exists():
        parts.append(f"""
= Centroid Matching

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    {_img(hung, base, "100%")}
    #align(center)[_Hungarian_]
  ],
  [
    {_img(greedy, base, "100%")}
    #align(center)[_Greedy_]
  ],
)
""")
    else:
        parts.append(_section_img(hung, "Centroid Matching (Hungarian)", base))
        parts.append(_section_img(greedy, "Centroid Matching (Greedy)", base))

    # Contingency & AUC heatmaps
    parts.append(
        _section_img(
            analysis_dir / "contingency_heatmap.png", "Contingency Heatmap", base
        )
    )

    # Optional ground-truth crosstab
    parts.append(
        _section_img(
            analysis_dir / "crosstab_heatmap.png", "Ground-Truth Crosstab Heatmap", base
        )
    )

    source = "\n".join(p for p in parts if p)
    return out_pdf if _compile(source, out_pdf) else None


# ─── Summary report ───────────────────────────────────────────────────────────

_SUMMARY_INT_ROWS = {"K", "Computed states", "Computed states > 1%", "Mapped states"}
_SUMMARY_EXCLUDE_ROWS = {"modularity_spatial_hvg__computed", "centroid_cosim__computed"}
_SUMMARY_MIN_BOLD_ROWS = {"per_state_perm_p"}


def _overview_table(overview_csv: Path) -> str:
    """
    Typst table for analysis_overview.csv with:
      - Integer rows (K, Computed States, …) printed without decimals
      - modularity_spatial_hvg__computed and centroid_cosim__computed rows omitted
      - All metric rows: maximum value per row printed bold
    """
    with open(overview_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return "#text[_(no data)_]"

    header = rows[0]
    n = len(header)
    data_rows = [r for r in rows[1:] if r and r[0] not in _SUMMARY_EXCLUDE_ROWS]

    lines = [
        "#table(",
        f"  columns: {n},",
        "  stroke: 0.5pt,",
        "  fill: (_, row) => if row == 0 { luma(220) } else if calc.odd(row) { luma(248) } else { white },",
        "  align: (col, _) => if col == 0 { left } else { right },",
        "  " + ", ".join(f"[*{_esc(h)}*]" for h in header) + ",",
    ]

    for row in data_rows:
        metric = row[0] if row else ""
        values = row[1:] if len(row) > 1 else []
        is_int_row = metric in _SUMMARY_INT_ROWS

        bold_idx = -1
        if not is_int_row and values:
            try:
                fvals = [float(v) if v else float("nan") for v in values]
                valid = [(i, v) for i, v in enumerate(fvals) if v == v]
                if valid:
                    pick = min if metric in _SUMMARY_MIN_BOLD_ROWS else max
                    bold_idx = pick(valid, key=lambda t: t[1])[0]
            except ValueError:
                pass

        cells = [f"[{_esc(metric)}]"]
        for i, v in enumerate(values):
            if is_int_row:
                try:
                    text = str(int(float(v)))
                except (ValueError, TypeError):
                    text = _esc(v)
            else:
                text = _fmt(v)
            cells.append(f"[*{text}*]" if i == bold_idx else f"[{text}]")

        lines.append("  " + ", ".join(cells) + ",")

    lines.append(")")
    return "\n".join(lines)


def _best_k_table(overview_csv: Path) -> str:
    """Build a Typst table showing the best run/K for each metric (all higher-is-better)."""
    with open(overview_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    if len(rows) < 2:
        return ""

    run_ids = rows[0][1:]
    k_row = next((r for r in rows[1:] if r and r[0] == "K"), None)
    k_vals = k_row[1:] if k_row else run_ids
    data_rows = [
        r for r in rows[1:] if r and r[0] != "K" and r[0] not in _SUMMARY_EXCLUDE_ROWS
    ]

    cells = ["  [*Metric*], [*Best run*], [*K*], [*Value*],"]
    for row in data_rows:
        metric = row[0]
        if metric in _SUMMARY_INT_ROWS:
            continue
        raw = row[1:]
        try:
            vals = [float(v) if v else float("nan") for v in raw]
        except ValueError:
            continue
        valid = [(i, v) for i, v in enumerate(vals) if v == v]  # drop NaN
        if not valid:
            continue
        best_i, best_v = max(valid, key=lambda t: t[1])
        best_run = run_ids[best_i] if best_i < len(run_ids) else "?"
        raw_k = k_vals[best_i] if best_i < len(k_vals) else "?"
        try:
            best_k = str(int(float(raw_k)))
        except (ValueError, TypeError):
            best_k = str(raw_k)
        cells.append(
            f"  [{_esc(metric)}], [{_esc(str(best_run))}], "
            f"[{best_k}], [{best_v:.4f}],"
        )

    if len(cells) == 1:  # only header row, nothing to show
        return ""

    return (
        "#table(\n"
        "  columns: 4,\n"
        "  stroke: 0.5pt,\n"
        "  fill: (_, row) => if row == 0 { luma(220) } else if calc.odd(row) { luma(248) } else { white },\n"
        "  align: (col, _) => if col == 0 { left } else { right },\n"
        + "\n".join(cells)
        + "\n"
        ")"
    )


def _umap_gallery(overview_csv: Path, pair_dir: Path) -> str:
    """Return Typst source for one UMAP image per run, stacked vertically."""
    with open(overview_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return ""

    run_ids = rows[0][1:]
    k_row = next((r for r in rows[1:] if r and r[0] == "K"), None)
    k_vals = k_row[1:] if k_row else ["?"] * len(run_ids)

    blocks: list[str] = []
    for run_id, k_val in zip(run_ids, k_vals):
        img_path = pair_dir / run_id / "analysis" / "umap_comparison.png"
        if not img_path.exists():
            continue
        k_label = int(float(k_val)) if k_val != "?" else run_id
        rel = _img_path(img_path, pair_dir)
        blocks.append(
            f'#text(weight: "bold")[K = {k_label} (run {run_id})]\n'
            f'#image("{rel}", width: 100%)\n'
            f"#v(0.5em)"
        )

    return "\n".join(blocks)


def generate_summary_report(pair_dir: Path) -> Path | None:
    """Generate a cross-K summary PDF and return its path."""
    pair_dir = Path(pair_dir)
    overview = pair_dir / "analysis_overview.csv"
    if not overview.exists():
        logger.warning("analysis_overview.csv not found — skipping summary report.")
        return None

    out_pdf = pair_dir / "summary_report.pdf"

    best_tbl = _best_k_table(overview)
    best_section = f"\n= Best Run per Metric\n\n{best_tbl}\n" if best_tbl else ""

    umap_gallery = _umap_gallery(overview, pair_dir)
    umap_section = (
        f"\n= UMAP Comparison per K\n\n{umap_gallery}\n" if umap_gallery else ""
    )

    source = _PAGE_SETUP + f"""

#align(center)[
  #text(size: 22pt, weight: "bold")[Analysis Summary Report]
  #v(0.3em)
  #text(size: 9pt, fill: luma(120))[Generated {_TODAY}]
]
#v(0.8em)
#line(length: 100%)
#v(0.5em)

= Overview: All Runs

#set text(size: 8.5pt)
{_overview_table(overview)}
#set text(size: 11pt)
{best_section}{umap_section}
"""

    return out_pdf if _compile(source, out_pdf) else None
