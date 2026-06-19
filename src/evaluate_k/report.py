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
import typst
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

    perm_p_col = header.index("perm_p") if "perm_p" in header else -1

    lines = [
        "#table(",
        f"  columns: {n},",
        "  stroke: 0.5pt,",
        "  fill: (_, row) => if row == 0 { luma(220) } else if calc.odd(row) { luma(248) } else { white },",
        "  align: (col, _) => if col == 0 { left } else { right },",
    ]

    # Header row
    lines.append("  " + ", ".join(f"[*{_esc(h)}*]" for h in header) + ",")

    cells_col = header.index("cells") if "cells" in header else -1
    spots_col = header.index("spots") if "spots" in header else -1
    n_ls_col = header.index("n_LS") if "n_LS" in header else -1

    def _keep(row: list[str]) -> bool:
        try:
            if cells_col >= 0 and int(float(row[cells_col])) <= 0:
                return False
            if spots_col >= 0 and int(float(row[spots_col])) <= 0:
                return False
            if n_ls_col >= 0 and int(float(row[n_ls_col])) <= 1:
                return False
        except (ValueError, IndexError):
            pass
        return True

    # Data rows
    for row in rows[1:]:
        if not _keep(row):
            continue
        col_fill: dict[int, str] = {}

        if perm_p_col >= 0 and perm_p_col < len(row):
            try:
                col_fill[perm_p_col] = _GREEN if float(row[perm_p_col]) < 0.33 else _RED
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


# ─── Per-K report ─────────────────────────────────────────────────────────────


def generate_per_k_report(
    analysis_dir: Path,
    K: int,
    run_id: str,
) -> Path | None:
    """Generate a PDF report for one analysis run and return its path."""
    analysis_dir = Path(analysis_dir)
    plots_dir = analysis_dir / "plots"
    data_dir = analysis_dir / "data"
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

    base = analysis_dir  # .typ lives here; image paths are relative to it

    _frac_note = (
        "#text(size: 8pt, fill: luma(140))"
        "[_Only states with cell-fraction > 1.0 % and spot-fraction > 1.0 % are shown._]\n"
    )

    # 1. UMAP comparison
    parts.append(
        _section_img(plots_dir / "umap_comparison.png", "UMAP Comparison", base)
    )

    # 2. Cell- and Spot-State Fractions (moved here, below UMAPs)
    sec = _section_img(
        plots_dir / "cell_state_fractions.png", "Cell- and Spot-State Fractions", base
    )
    if sec:
        parts.append(sec + _frac_note)

    # 3. Spatial cell-state plot
    parts.append(
        _section_img(
            plots_dir / "spatial_cell_states.png",
            "Spatial Distribution of Cell States",
            base,
        )
    )

    # 4. Cell-state profiles
    sec = _section_img(
        plots_dir / "cell_state_profiles.png", "Cell-State Profiles", base
    )
    if sec:
        parts.append(sec + _frac_note)

    # Contingency heatmap
    parts.append(
        _section_img(plots_dir / "contingency_heatmap.png", "Contingency Heatmap", base)
    )

    # Per-state sub-cluster analysis
    substate_csv = data_dir / "substate_metrics.csv"
    if substate_csv.exists():
        parts.append(f"""
= Per-State Sub-cluster Analysis

#set text(size: 7pt)
{_substate_metrics_table(substate_csv)}
#set text(size: 11pt)
""")

    # Optional ground-truth crosstab
    parts.append(
        _section_img(
            plots_dir / "crosstab_heatmap.png", "Ground-Truth Crosstab Heatmap", base
        )
    )

    source = "\n".join(p for p in parts if p)
    return out_pdf if _compile(source, out_pdf) else None


# ─── Summary report ───────────────────────────────────────────────────────────

_SUMMARY_INT_ROWS = {"K", "Computed states", "Computed states > 1%", "Mapped states"}
_SUMMARY_EXCLUDE_ROWS = {
    "modularity_spatial_hvg__computed",
    "centroid_cosim__computed",
    "silhouette__computed",
    "dunn_index__computed",
}
_SUMMARY_ROW_LABELS = {
    "per_state_perm_p": "Substate p-value",
    "modularity__computed": "Modularity",
    "contingency_score": "Contingency",
    "median_cossim_gene": "Cossim (genewise)",
    "median_cossim_spot": "Cossim (spotwise)",
}
_SUMMARY_MIN_BOLD_ROWS = {"per_state_perm_p", "Substate p-value"}


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
        label = _SUMMARY_ROW_LABELS.get(metric, metric)
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

        cells = [f"[{_esc(label)}]"]
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
        img_path = pair_dir / run_id / "analysis" / "plots" / "umap_comparison.png"
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
{umap_section}
"""

    return out_pdf if _compile(source, out_pdf) else None
