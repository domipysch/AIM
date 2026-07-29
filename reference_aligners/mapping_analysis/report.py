"""PDF report generation via Typst for reference-aligner mapping analysis.

Mirrors the pattern used by analysis/report.py:
build a Typst source string, write it to a temp .typ file, compile via the
`typst` Python bindings, then discard the .typ (only the PDF persists).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import typst

logger = logging.getLogger(__name__)

_TODAY = date.today().isoformat()

_PAGE_SETUP = """\
#set page(paper: "a4", margin: (top: 2cm, bottom: 2cm, left: 2.5cm, right: 2.5cm))
#set text(size: 11pt)
#set heading(numbering: "1.")
"""


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


def _img_path(p: Path, base: Path) -> str:
    """Return a Typst-safe image path relative to *base* (the .typ file's directory)."""
    try:
        rel = p.resolve().relative_to(base.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return p.resolve().as_uri()


def _img(p: Path, base: Path, width: str = "100%") -> str:
    return f'#image("{_img_path(p, base)}", width: {width})'


def _section_img(p: Path, title: str, base: Path, width: str = "100%") -> str:
    """Return a Typst heading + image section, or '' if the file is absent."""
    if not p.exists():
        return ""
    return f"\n= {title}\n\n{_img(p, base, width)}\n"


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


def _cossim_rows(cossim_summary: dict[str, dict]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for label, vals in cossim_summary.items():
        g, s = vals.get("median_gene"), vals.get("median_spot")
        rows.append(
            (f"{label} — gene-wise median", f"{g:.4f}" if g is not None else "n/a")
        )
        rows.append(
            (f"{label} — spot-wise median", f"{s:.4f}" if s is not None else "n/a")
        )
    return rows


def _compile(source: str, out_pdf: Path) -> Path | None:
    typ_file = out_pdf.with_suffix(".typ")
    typ_file.write_text(source, encoding="utf-8")
    try:
        pdf_bytes = typst.compile(str(typ_file), format="pdf")
        out_pdf.write_bytes(pdf_bytes)
        logger.info("Report → %s", out_pdf)
        return out_pdf
    except Exception as e:
        logger.error("typst compile failed: %s", e)
        return None
    finally:
        typ_file.unlink(missing_ok=True)


def generate_report(
    analysis_dir: Path,
    mapping_source: str,
    onehot_summary: dict,
    cell_type_counts: dict[str, int],
    cossim_summary: dict[str, dict],
) -> Path | None:
    """Build and compile the mapping-analysis PDF report.

    Args:
        analysis_dir: Directory containing plots/ (as written by analyze_mapping);
                       report.pdf is written directly inside it.
        mapping_source: Human-readable label for the analyzed mapping (e.g. the
                        mapping folder path), shown in the report header.
        onehot_summary: Return value of metrics.onehot_metrics().
        cell_type_counts: cell type name -> number of sc cells.
        cossim_summary: combo label -> {"median_gene": float|None, "median_spot": float|None}.
    """
    analysis_dir = Path(analysis_dir)
    plots_dir = analysis_dir / "plots"
    out_pdf = analysis_dir / "report.pdf"
    base = analysis_dir

    parts: list[str] = [_PAGE_SETUP]
    parts.append(f"""
#align(center)[
  #text(size: 22pt, weight: "bold")[Reference Aligner Mapping Analysis]
  #v(0.3em)
  #text(size: 12pt)[{_esc(mapping_source)}]
  #v(0.3em)
  #text(size: 9pt, fill: luma(120))[Generated {_TODAY}]
]
#v(0.8em)
#line(length: 100%)
#v(0.5em)
""")

    # 1. Mapping sharpness ("how one-hot")
    sec = _section_img(
        plots_dir / "onehot_distribution.png",
        'Mapping Sharpness ("How One-Hot")',
        base,
    )
    if sec:
        s = onehot_summary["summary"]
        rows = [
            ("Spots", str(onehot_summary["n_rows"])),
            ("Cell types", str(onehot_summary["n_cols"])),
            ("Max-prob mean", f"{s['max_prob']['mean']:.4f}"),
            ("Max-prob median", f"{s['max_prob']['median']:.4f}"),
            ("Gini impurity mean", f"{s['gini_impurity']['mean']:.4f}"),
            ("Entropy mean", f"{s['entropy']['mean']:.4f}"),
            ("Spots with max-prob > 0.5", f"{s['frac_max_prob_above_0.5']:.1%}"),
            ("Spots with max-prob > 0.9", f"{s['frac_max_prob_above_0.9']:.1%}"),
            ("Spots with max-prob > 0.99", f"{s['frac_max_prob_above_0.99']:.1%}"),
        ]
        parts.append(sec + "\n" + _two_col_table(rows) + "\n")
        parts.append(
            _img(plots_dir / "onehot_threshold_fractions.png", base)
            if (plots_dir / "onehot_threshold_fractions.png").exists()
            else ""
        )

    # 2. SC UMAP and spatial hard-mapping side by side (same cell-type colours)
    umap_png = plots_dir / "sc_umap_celltype.png"
    spatial_png = plots_dir / "spatial_hard_celltypes.png"
    if umap_png.exists() and spatial_png.exists():
        parts.append(
            "\n= Cell Types — scRNA UMAP vs Spatial Hard-Mapping\n\n"
            "#grid(columns: 2, column-gutter: 8pt, align: horizon,\n"
            f"  [{_img(umap_png, base)}],\n"
            f"  [{_img(spatial_png, base)}],\n"
            ")\n"
        )
        if cell_type_counts:
            rows = [(ct, str(n)) for ct, n in cell_type_counts.items()]
            parts.append(
                "\n" + _two_col_table(rows, header=("Cell type", "Cells")) + "\n"
            )
    else:
        sec = _section_img(umap_png, "scRNA Cell Types (UMAP)", base)
        if sec:
            rows = [(ct, str(n)) for ct, n in cell_type_counts.items()]
            parts.append(
                sec + "\n" + _two_col_table(rows, header=("Cell type", "Cells")) + "\n"
            )
        parts.append(
            _section_img(
                spatial_png, "Spatial Distribution — Hard-Mapped Cell Types", base
            )
        )

    # 3. Cell-type centroid z-scores
    parts.append(
        _section_img(
            plots_dir / "celltype_centroid_zscores.png",
            "Cell-Type Marker Gene Profiles",
            base,
        )
    )

    # 5. Reconstruction cosine similarity
    if cossim_summary:
        boxplot_path = plots_dir / "cossim_boxplots.png"
        boxplot_block = (
            f"\n{_img(boxplot_path, base)}\n" if boxplot_path.exists() else ""
        )
        parts.append(f"""
= Reconstruction Cosine Similarity

Predicted spot expression is reconstructed as `mapping @ cell_type_centroids`,
restricted to genes shared between the scRNA and ST data, and compared
against the observed ST expression. "soft" uses the mapping as given; "hard"
uses its row-wise argmax one-hot version. "raw" compares raw counts; "norm"
compares total-count-normalized + log1p-transformed data.

{_two_col_table(_cossim_rows(cossim_summary), header=("Combo", "Median cossim"))}
{boxplot_block}""")

    source = "\n".join(p for p in parts if p)
    return _compile(source, out_pdf)
