"""
PDF report generation via Typst for the pre-alignment compatibility check.

Public API
----------
generate_pre_check_report(output_dir) -> Path | None

Reads the files written by run_pre_check() from *output_dir* and compiles a
single PDF.  Returns None (with a warning) when typst is not on PATH.
The .typ source file is kept beside the PDF for debugging.
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_TODAY = date.today().isoformat()

_PAGE_SETUP = """\
#set page(paper: "a4", margin: (top: 2cm, bottom: 2cm, left: 2.5cm, right: 2.5cm))
#set text(size: 11pt)
#set heading(numbering: "1.")
"""


# ─── Helpers (same pattern as evaluate_k/report.py) ──────────────────────────


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("#", "\\#")
        .replace("@", "\\@")
        .replace("_", "\\_")
    )


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return _esc(str(v))


def _img_path(p: Path, base: Path) -> str:
    """Path relative to *base* (the .typ file's directory), forward slashes."""
    try:
        rel = p.resolve().relative_to(base.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return p.resolve().as_uri()


def _img(p: Path, base: Path, width: str = "100%") -> str:
    return f'#image("{_img_path(p, base)}", width: {width})'


def _section_img(p: Path, title: str, base: Path, width: str = "100%") -> str:
    if not p.exists():
        return ""
    return f"\n= {title}\n\n{_img(p, base, width)}\n"


def _compile(source: str, out_pdf: Path) -> bool:
    typ_file = out_pdf.with_suffix(".typ")
    typ_file.write_text(source, encoding="utf-8")
    try:
        r = subprocess.run(
            ["typst", "compile", str(typ_file), str(out_pdf)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if r.returncode != 0:
            logger.error("typst compile failed:\n%s", r.stderr)
            return False
        logger.info("Pre-check report → %s", out_pdf)
        return True
    except FileNotFoundError:
        logger.warning("typst not found on PATH — skipping PDF report.")
        typ_file.unlink(missing_ok=True)
        return False
    except subprocess.TimeoutExpired:
        logger.error("typst compile timed out for %s", out_pdf)
        return False


# ─── Table builders ───────────────────────────────────────────────────────────


def _two_col_table(
    rows: list[tuple[str, str]], header: tuple[str, str] = ("Metric", "Value")
) -> str:
    """Simple two-column Typst table from a list of (label, value) tuples."""
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


def _three_col_table(
    rows: list[tuple[str, str, str]],
    header: tuple[str, str, str] = ("Metric", "SC", "ST"),
) -> str:
    """Three-column Typst table (metric, SC value, ST value)."""
    lines = [
        "#table(",
        "  columns: 3,",
        "  stroke: 0.5pt,",
        "  fill: (_, row) => if row == 0 { luma(220) } else if calc.odd(row) { luma(248) } else { white },",
        "  align: (col, _) => if col == 0 { left } else { right },",
        f"  [*{_esc(header[0])}*], [*{_esc(header[1])}*], [*{_esc(header[2])}*],",
    ]
    for label, v_sc, v_st in rows:
        lines.append(f"  [{_esc(label)}], [{_esc(str(v_sc))}], [{_esc(str(v_st))}],")
    lines.append(")")
    return "\n".join(lines)


# ─── Section builders ─────────────────────────────────────────────────────────


def _dataset_overview_section(summary: dict) -> str:
    def _pct(v):
        return f"{v:.2f} %"

    rows = [
        (
            "Observations",
            summary.get("n_cells_sc", "—"),
            summary.get("n_spots_st", "—"),
        ),
        (
            "Genes (total)",
            summary.get("n_genes_sc", "—"),
            summary.get("n_genes_st", "—"),
        ),
        (
            "Shared genes",
            summary.get("n_shared_genes", "—"),
            summary.get("n_shared_genes", "—"),
        ),
        (
            "Shared of total",
            _pct(summary.get("pct_shared_of_sc", 0)),
            _pct(summary.get("pct_shared_of_st", 0)),
        ),
        (
            "Library size (mean)",
            summary.get("sc_libsize_mean", "—"),
            summary.get("st_libsize_mean", "—"),
        ),
        (
            "Library size (median)",
            summary.get("sc_libsize_median", "—"),
            summary.get("st_libsize_median", "—"),
        ),
        (
            "Library size (std)",
            summary.get("sc_libsize_std", "—"),
            summary.get("st_libsize_std", "—"),
        ),
        ("Sparsity", summary.get("sc_sparsity", "—"), summary.get("st_sparsity", "—")),
        (
            "Leiden clusters",
            summary.get("n_clusters_sc", "—"),
            summary.get("n_clusters_st", "—"),
        ),
        (
            "Silhouette score",
            summary.get("silhouette_sc", "—"),
            summary.get("silhouette_st", "—"),
        ),
        ("Dunn index", summary.get("dunn_sc", "—"), summary.get("dunn_st", "—")),
    ]
    return f"""
= Dataset Overview

{ _three_col_table(rows, header=("", "SC", "ST")) }
"""


def _metrics_section(results: dict) -> str:
    rows = [
        ("Shared genes", results.get("n_shared_genes", "—")),
        ("SC Leiden clusters", results.get("n_clusters_sc", "—")),
        ("ST Leiden clusters", results.get("n_clusters_st", "—")),
        (
            "Variance rank Spearman ρ",
            f"{results.get('variance_rank_spearman', float('nan')):.4f}",
        ),
        (
            "Centroid cosine sim (Hungarian)",
            f"{results.get('centroid_cosine_sim', float('nan')):.4f}",
        ),
        (
            "Centroid cosine sim (Greedy)",
            f"{results.get('greedy_cosine_sim', float('nan')):.4f}",
        ),
        (
            "Top-gene Jaccard (top 5)",
            f"{results.get('top_gene_jaccard_top5', float('nan')):.4f}",
        ),
        (
            "Top-gene Jaccard (top 10)",
            f"{results.get('top_gene_jaccard_top10', float('nan')):.4f}",
        ),
        (
            "Top-gene Jaccard (top 20)",
            f"{results.get('top_gene_jaccard_top20', float('nan')):.4f}",
        ),
    ]
    pt = results.get("permutation_test")
    if pt:
        rows += [
            ("Permutation test z-score", f"{pt.get('z_score', float('nan')):.3f}"),
            ("Permutation test p-value", f"{pt.get('p_value', float('nan')):.4f}"),
            ("Permutation null mean", f"{pt.get('null_mean', float('nan')):.4f}"),
        ]
    return f"""
= Compatibility Metrics

{_two_col_table(rows)}
"""


# ─── Main entry point ─────────────────────────────────────────────────────────


def generate_pre_check_report(output_dir: Path) -> Path | None:
    """Compile a PDF report from the files written by run_pre_check()."""
    output_dir = Path(output_dir)
    out_pdf = output_dir / "pre_check_report.pdf"
    plots_dir = output_dir / "plots"
    base = output_dir  # .typ lives here; all paths are relative to it

    # ── Load JSON data ────────────────────────────────────────────────────────
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.json"

    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    sc_name = output_dir.name  # best we can do without the original adata

    # ── Assemble source ───────────────────────────────────────────────────────
    parts: list[str] = [_PAGE_SETUP]

    # Title
    parts.append(f"""
#align(center)[
  #text(size: 22pt, weight: "bold")[Pre-Alignment Compatibility Report]
  #v(0.3em)
  #text(size: 12pt)[{_esc(sc_name)}]
  #v(0.3em)
  #text(size: 9pt, fill: luma(120))[Generated {_TODAY}]
]
#v(0.8em)
#line(length: 100%)
#v(0.5em)
""")

    # Dataset overview + metrics tables
    if summary:
        parts.append(_dataset_overview_section(summary))
    if results:
        parts.append(_metrics_section(results))

    # Gene expression concordance
    parts.append(
        f"""
= Gene Expression Concordance

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    {_img(plots_dir / "mean_scatter.png", base, "100%")}
    #align(center)[_Per-gene mean expression_]
  ],
  [
    {_img(plots_dir / "dropout_scatter.png", base, "100%")}
    #align(center)[_Per-gene dropout rate_]
  ],
)
"""
        if (plots_dir / "mean_scatter.png").exists()
        else ""
    )

    parts.append(
        _section_img(
            plots_dir / "variance_rank_scatter.png", "Variance Rank Concordance", base
        )
    )

    # Library sizes
    parts.append(
        _section_img(
            plots_dir / "library_sizes.png", "Library Size Distributions", base
        )
    )

    # UMAP + spatial
    parts.append(
        _section_img(plots_dir / "umap.png", "UMAP Embeddings (SC and ST)", base)
    )
    parts.append(
        _section_img(
            plots_dir / "spatial_clusters.png", "ST Spatial Cluster Layout", base
        )
    )

    # HVG heatmap
    parts.append(
        _section_img(
            plots_dir / "hvg_heatmap.png",
            "Cluster-Mean Expression — Shared Genes",
            base,
        )
    )

    # Centroid similarity
    hung = plots_dir / "centroid_heatmap.png"
    greedy = plots_dir / "centroid_heatmap_greedy.png"
    if hung.exists() and greedy.exists():
        parts.append(f"""
= Centroid Cosine Similarity

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    {_img(hung, base, "100%")}
    #align(center)[_Hungarian matching_]
  ],
  [
    {_img(greedy, base, "100%")}
    #align(center)[_Greedy best match_]
  ],
)
""")
    else:
        parts.append(_section_img(hung, "Centroid Cosine Similarity (Hungarian)", base))
        parts.append(_section_img(greedy, "Centroid Cosine Similarity (Greedy)", base))

    # Matched cluster overview
    parts.append(
        _section_img(
            plots_dir / "matched_clusters" / "all_matches_overview.png",
            "Matched Cluster Pairs — Expression Overview",
            base,
        )
    )

    # Permutation test (optional)
    parts.append(
        _section_img(
            plots_dir / "permutation_null.png",
            "Permutation Test — Null Distribution",
            base,
        )
    )

    source = "\n".join(p for p in parts if p)
    return out_pdf if _compile(source, out_pdf) else None
