import shutil
from pathlib import Path
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import json
from anndata import AnnData
from .utils.dataset_query import get_z_real_and_predicted_data_only_shared_genes
from ..utils.io import load_st_adata
from .utils.distance_metrics import cosine_similarity

logger = logging.getLogger(__name__)


def compute_metrics_per_gene(
    adata_z, adata_predicted_z, save_cossim_json: Path | None = None
) -> None:
    """
    Compute cosine similarity (or other distance metrics) per gene and write results directly into
    adata_predicted_z.var['cossim'].

    Optional: if save_cossim_json is provided (Path), the per-gene cossim values
    are also written to a JSON file (gene -> float).
    """

    # initialize column with NaN if not already present
    adata_predicted_z.var["cossim"] = np.nan
    # adata_predicted_z.var['sqrt_cossim'] = np.nan
    # adata_predicted_z.var['eucl'] = np.nan
    # adata_predicted_z.var['rmse'] = np.nan
    # adata_predicted_z.var['mae'] = np.nan
    # adata_predicted_z.var['canberra'] = np.nan
    # adata_predicted_z.var['pearson'] = np.nan
    # adata_predicted_z.var['bray_curtis'] = np.nan
    # adata_predicted_z.var['aitchison'] = np.nan
    # adata_predicted_z.var['kl'] = np.nan
    # adata_predicted_z.var['js'] = np.nan
    # adata_predicted_z.var['hellinger'] = np.nan
    # adata_predicted_z.var['bhat'] = np.nan
    # adata_predicted_z.var['tv'] = np.nan

    # Collect cossim values for optional export
    cossim_dict = {}

    counter = 0
    for gene in adata_predicted_z.var_names:

        # Retrieve vectors (AnnData slicing may return 2D arrays)
        vec_z = adata_z[:, gene].X.toarray().ravel()
        vec_pred = adata_predicted_z[:, gene].X.toarray().ravel()

        # Compute and store values in local variables before assignment (useful for export)
        val_cossim = float(cosine_similarity(vec_z, vec_pred))
        # val_sqrt_cossim = float(sqrt_cosine_similarity(vec_z, vec_pred))
        # val_eucl = float(euclidean_l2(vec_z, vec_pred))
        # val_rmse = float(rmse(vec_z, vec_pred))
        # val_mae = float(mae_l1(vec_z, vec_pred))
        # val_canberra = float(canberra(vec_z, vec_pred))
        # val_pearson = float(pearson_distance(vec_z, vec_pred))
        # val_bray = float(bray_curtis_distance(vec_z, vec_pred))
        # val_aitchison = float(aitchison_distance(vec_z, vec_pred))
        # val_kl = float(kl_divergence(vec_z, vec_pred))
        # val_js = float(jensen_shannon_distance(vec_z, vec_pred))
        # val_hell = float(hellinger_distance(vec_z, vec_pred))
        # val_bhat = float(bhattacharyya_distance(vec_z, vec_pred))
        # val_tv = float(total_variation(vec_z, vec_pred))

        # Store directly in adata_predicted_z.var
        adata_predicted_z.var.at[gene, "cossim"] = val_cossim
        # adata_predicted_z.var.at[gene, 'sqrt_cossim'] = val_sqrt_cossim
        # adata_predicted_z.var.at[gene, 'eucl'] = val_eucl
        # adata_predicted_z.var.at[gene, 'rmse'] = val_rmse
        # adata_predicted_z.var.at[gene, 'mae'] = val_mae
        # adata_predicted_z.var.at[gene, 'canberra'] = val_canberra
        # adata_predicted_z.var.at[gene, 'pearson'] = val_pearson
        # adata_predicted_z.var.at[gene, 'bray_curtis'] = val_bray
        # adata_predicted_z.var.at[gene, 'aitchison'] = val_aitchison
        # adata_predicted_z.var.at[gene, 'kl'] = val_kl
        # adata_predicted_z.var.at[gene, 'js'] = val_js
        # adata_predicted_z.var.at[gene, 'hellinger'] = val_hell
        # adata_predicted_z.var.at[gene, 'bhat'] = val_bhat
        # adata_predicted_z.var.at[gene, 'tv'] = val_tv

        # Collect for export
        cossim_dict[gene] = val_cossim

        counter += 1
        if counter % 1000 == 0:
            logging.info(f"Processed {counter}/{adata_predicted_z.n_vars} genes.")

    # Optional: save cossim per gene as JSON
    if save_cossim_json is not None:
        save_cossim_json.parent.mkdir(parents=True, exist_ok=True)
        values = list(cossim_dict.values())
        output = {
            "median": float(np.median(values)) if values else None,
            "values": cossim_dict,
        }
        with save_cossim_json.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)


def compute_metrics_per_spot(
    adata_z, adata_predicted_z, save_cossim_json: Path | None = None
) -> None:
    """
    Compute cosine similarity (or other distance metrics) per spot and write results directly into
    adata_predicted_z.obs['cossim'].
    """

    # initialize obs columns with NaN if not already present
    adata_predicted_z.obs["cossim"] = np.nan
    # adata_predicted_z.obs['sqrt_cossim'] = np.nan
    # adata_predicted_z.obs['eucl'] = np.nan
    # adata_predicted_z.obs['rmse'] = np.nan
    # adata_predicted_z.obs['mae'] = np.nan
    # adata_predicted_z.obs['canberra'] = np.nan
    # adata_predicted_z.obs['pearson'] = np.nan
    # adata_predicted_z.obs['bray_curtis'] = np.nan
    # adata_predicted_z.obs['aitchison'] = np.nan
    # adata_predicted_z.obs['kl'] = np.nan
    # adata_predicted_z.obs['js'] = np.nan
    # adata_predicted_z.obs['hellinger'] = np.nan
    # adata_predicted_z.obs['bhat'] = np.nan
    # adata_predicted_z.obs['tv'] = np.nan

    # Collect cossim values for optional export
    cossim_dict = {}

    counter = 0
    for spot in adata_predicted_z.obs_names:
        # Retrieve vectors (AnnData slicing may return 2D arrays)
        vec_z = adata_z[spot, :].X.toarray().ravel()
        vec_pred = adata_predicted_z[spot, :].X.toarray().ravel()

        # Store metrics directly in adata_predicted_z.obs
        val_cossim = cosine_similarity(vec_z, vec_pred)
        adata_predicted_z.obs.at[spot, "cossim"] = val_cossim
        # adata_predicted_z.obs.at[spot, 'sqrt_cossim'] = sqrt_cosine_similarity(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'eucl'] = euclidean_l2(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'rmse'] = rmse(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'mae'] = mae_l1(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'canberra'] = canberra(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'pearson'] = pearson_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'bray_curtis'] = bray_curtis_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'aitchison'] = aitchison_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'kl'] = kl_divergence(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'js'] = jensen_shannon_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'hellinger'] = hellinger_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'bhat'] = bhattacharyya_distance(vec_z, vec_pred)
        # adata_predicted_z.obs.at[spot, 'tv'] = total_variation(vec_z, vec_pred)
        cossim_dict[spot] = val_cossim

        counter += 1
        if counter % 1000 == 0:
            logging.info(f"Processed {counter}/{adata_predicted_z.n_obs} spots.")

    # Optional: save cossim per spot as JSON
    if save_cossim_json is not None:
        save_cossim_json.parent.mkdir(parents=True, exist_ok=True)
        values = list(cossim_dict.values())
        output = {
            "median": float(np.median(values)) if values else None,
            "values": cossim_dict,
        }
        with save_cossim_json.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)


def generate_box_plot_metrics_per_gene(
    adata_predicted_z,
    output_folder: Path | None = None,
    columns: list[str] | None = None,
) -> None:
    """
    Generate boxplots for numeric metrics in adata_predicted_z.var.
    - All metrics whose (non-NaN) values lie within [0,1] are plotted together in one figure.
    - All other metrics get individual figures (one boxplot dimension per figure).
    - output_folder is optional; if omitted, plots are shown instead of saved.
    - columns: list of column names to plot; defaults to ['cossim'].
    """
    if columns is None:
        columns = ["cossim"]
    df_var = adata_predicted_z.var
    numeric = df_var.select_dtypes(include=[np.number])
    numeric = numeric[[c for c in columns if c in numeric.columns]]
    if numeric.shape[1] == 0:
        raise ValueError("No numeric metrics found in adata_predicted_z.var.")

    tol = 1e-8
    unit_cols = []
    other_cols = []
    col_data = {}

    # Classify columns
    for col in numeric.columns:
        vals = numeric[col].dropna().values
        if vals.size == 0:
            # skip empty columns (treat as 'other' but without plotting data)
            other_cols.append(col)
            col_data[col] = vals
            continue
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        col_data[col] = vals
        if vmin >= -tol and vmax <= 1.0 + tol:
            unit_cols.append(col)
        else:
            other_cols.append(col)

    # Combined plot for all [0,1] metrics (if any)
    if len(unit_cols) > 0:
        data_unit = [col_data[c] for c in unit_cols]
        n = len(unit_cols)
        width = max(6, n * 0.4)
        fig, ax = plt.subplots(1, 1, figsize=(width, 5), constrained_layout=True)
        bp = ax.boxplot(
            data_unit,
            tick_labels=unit_cols,
            patch_artist=True,
            medianprops=dict(color="black"),
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#b2df8a")
            patch.set_edgecolor("black")
            patch.set_alpha(0.9)
        ax.set_title("Metrics in [0,1]")
        ax.set_ylabel("Value")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

        if output_folder:
            plt.savefig(output_folder / f"o2_0-1_metrics.pdf", bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)

    # Individual plots for all other metrics
    for col in other_cols:
        vals = col_data.get(col, np.array([]))
        # If truly no values, draw empty plot with a message
        fig, ax = plt.subplots(1, 1, figsize=(4, 5), constrained_layout=True)
        if vals.size == 0:
            ax.text(0.5, 0.5, f"No data for {col}", ha="center", va="center")
            ax.set_title(col)
            ax.set_xticks([])
        else:
            bp = ax.boxplot(
                vals,
                tick_labels=[col],
                patch_artist=True,
                medianprops=dict(color="black"),
            )
            box_color = "#a6cee3"
            for patch in bp["boxes"]:
                patch.set_facecolor(box_color)
                patch.set_edgecolor("black")
                patch.set_alpha(0.9)
            ax.set_ylabel("Value")
            ax.set_title(col)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

        if output_folder:
            plt.savefig(output_folder / f"o2_{col}.pdf", bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)


def generate_box_plot_metrics_per_spot(
    adata_predicted_z,
    output_folder: Path | None = None,
    columns: list[str] | None = None,
) -> None:
    """
    Generate boxplots for numeric metrics in adata_predicted_z.obs.
    - All metrics whose (non-NaN) values lie within [0,1] are plotted together in one figure.
    - All other metrics get individual figures (one boxplot dimension per figure).
    - output_folder (optional): if provided, plots are saved there as PNGs; otherwise they are shown.
    - columns: list of column names to plot; defaults to ['cossim'].
    """
    if columns is None:
        columns = ["cossim"]

    df_obs = adata_predicted_z.obs
    numeric = df_obs.select_dtypes(include=[np.number])
    numeric = numeric[[c for c in columns if c in numeric.columns]]
    if numeric.shape[1] == 0:
        raise ValueError("No numeric metrics found in adata_predicted_z.obs.")

    tol = 1e-8
    unit_cols = []
    other_cols = []
    col_data = {}

    # Classify columns
    for col in numeric.columns:
        vals = numeric[col].dropna().values
        if vals.size == 0:
            other_cols.append(col)
            col_data[col] = vals
            continue
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        col_data[col] = vals
        if vmin >= -tol and vmax <= 1.0 + tol:
            unit_cols.append(col)
        else:
            other_cols.append(col)

    # Combined plot for all [0,1] metrics (if any)
    if len(unit_cols) > 0:
        data_unit = [col_data[c] for c in unit_cols]
        n = len(unit_cols)
        width = max(6, n * 0.4)
        fig, ax = plt.subplots(1, 1, figsize=(width, 5), constrained_layout=True)
        bp = ax.boxplot(
            data_unit,
            tick_labels=unit_cols,
            patch_artist=True,
            medianprops=dict(color="black"),
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#b2df8a")
            patch.set_edgecolor("black")
            patch.set_alpha(0.9)
        ax.set_title("Spot metrics in [0,1]")
        ax.set_ylabel("Value")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_folder / f"o2_spots_0-1_metrics.pdf", bbox_inches="tight"
            )
        else:
            plt.show()
        plt.close(fig)

    # Individual plots for all other metrics
    for col in other_cols:
        vals = col_data.get(col, np.array([]))
        fig, ax = plt.subplots(1, 1, figsize=(4, 5), constrained_layout=True)
        if vals.size == 0:
            ax.text(0.5, 0.5, f"No data for {col}", ha="center", va="center")
            ax.set_title(col)
            ax.set_xticks([])
        else:
            bp = ax.boxplot(
                vals,
                tick_labels=[col],
                patch_artist=True,
                medianprops=dict(color="black"),
            )
            box_color = "#a6cee3"
            for patch in bp["boxes"]:
                patch.set_facecolor(box_color)
                patch.set_edgecolor("black")
                patch.set_alpha(0.9)
            ax.set_ylabel("Value")
            ax.set_title(col)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_folder / f"o2_spots_{col}.pdf", bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)


def generate_gene_spatial_distribution_plot(
    adata_z, adata_predicted_z, gene_name: str, output_path: Path
) -> None:
    """
    Generate a spatial distribution plot for a given gene:
    - Left panel: real Z data (adata_z)
    - Right panel: predicted Z data (adata_predicted_z)

    Change: all spots are now always drawn using the colormap (including zeros),
    i.e., there are no light-gray 'uncolored' points anymore.
    """
    # Ensure the gene exists
    if (
        gene_name not in adata_z.var_names
        or gene_name not in adata_predicted_z.var_names
    ):
        raise ValueError(
            f"Gene '{gene_name}' nicht in beiden AnnData-Objekten gefunden."
        )

    # Try to read cossim from adata_predicted_z.var (if available)
    cossim_val = float(adata_predicted_z.var.at[gene_name, "cossim"])

    # Expression vectors per spot (1D) - robust against sparse/dense matrices
    def vec_flat(adata, g):
        X = adata[:, g].X
        return X.toarray().ravel() if hasattr(X, "toarray") else np.asarray(X).ravel()

    vec_z = vec_flat(adata_z, gene_name).astype(float)
    vec_pred = vec_flat(adata_predicted_z, gene_name).astype(float)

    # Coordinates (assume adata.obsm['coords'] exists and matches obs_names order)
    if "coords" not in adata_z.obsm or "coords" not in adata_predicted_z.obsm:
        raise ValueError(
            "Räumliche Koordinaten (obsm['coords']) fehlen in einem der AnnData-Objekte."
        )
    coords_z = np.asarray(adata_z.obsm["coords"])
    coords_pred = np.asarray(adata_predicted_z.obsm["coords"])

    # Check shapes
    if coords_z.shape[0] != vec_z.shape[0] or coords_pred.shape[0] != vec_pred.shape[0]:
        raise ValueError(
            "Anzahl der Spots stimmt nicht mit der Länge der Expressionsvektoren überein."
        )

    cmap = plt.cm.get_cmap("viridis")

    # --- increased font sizes ---
    suptitle_fs = 14
    title_fs = 13
    label_fs = 12
    tick_fs = 11
    cbar_label_fs = 11
    # -----------------------------

    # Create 1x2 subplot figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    # New helper: draw ALL spots with the colormap (including zero values)
    def _draw_panel(ax, coords, values, title):
        vals = np.asarray(values).astype(float)
        # Local min/max (vmin can be 0)
        vmin_local = np.nanmin(vals) if np.any(np.isfinite(vals)) else 0.0
        vmax_local = float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else 1.0
        if not np.isfinite(vmin_local):
            vmin_local = 0.0
        if not np.isfinite(vmax_local) or vmax_local <= vmin_local:
            vmax_local = vmin_local + 1.0

        norm_local = plt.Normalize(vmin=vmin_local, vmax=vmax_local)
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=vals,
            cmap=cmap,
            norm=norm_local,
            s=40,
            edgecolors="k",
            linewidths=0.2,
        )
        ax.set_title(title, fontsize=title_fs)
        ax.set_xlabel("x", fontsize=label_fs)
        ax.set_ylabel("y", fontsize=label_fs)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
        ax.set_aspect("equal", adjustable="box")
        return sc, norm_local, vmin_local, vmax_local

    sc1, norm1, vmin1, v1 = _draw_panel(
        axes[0], coords_z, vec_z, f"Input data Z - {gene_name}"
    )
    sc2, norm2, vmin2, v2 = _draw_panel(
        axes[1], coords_pred, vec_pred, f"Predicted data Z' - {gene_name}"
    )

    # Separate colorbars: one colorbar per axis (local scale)
    if sc1 is not None:
        cbar1 = fig.colorbar(
            sc1, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.02
        )
        cbar1.set_label(
            f"Expression (counts), range [{vmin1:.2f}, {v1:.2f}]",
            fontsize=cbar_label_fs,
        )
        cbar1.ax.tick_params(labelsize=tick_fs)
    if sc2 is not None:
        cbar2 = fig.colorbar(
            sc2, ax=axes[1], orientation="vertical", fraction=0.046, pad=0.02
        )
        cbar2.set_label(
            f"Expression (counts), range [{vmin2:.2f}, {v2:.2f}]",
            fontsize=cbar_label_fs,
        )
        cbar2.ax.tick_params(labelsize=tick_fs)

    # Show cosine similarity value prominently as suptitle (if available)
    fig.suptitle(
        f"Gene {gene_name} — cosine similarity: {cossim_val:.3f}", fontsize=suptitle_fs
    )

    # Save or show (ensure target directory exists)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150)
    else:
        plt.show()
    plt.close(fig)


def generate_spatial_distribution_plots_for_some_genes(
    adata_z, adata_predicted_z, metrics_folder_spatial_per_gene: Path
) -> None:
    """
    Pick a few genes based on cosine similarity and generate spatial distribution plots for them.

    Args:
        adata_z:
        adata_predicted_z:
        metrics_folder_spatial_per_gene:

    Returns:

    """
    # Compute spatial distribution plots for 9 genes: 3 high / 3 medium / 3 low cossim (random within each bucket)
    np.random.seed(42)
    # safe cossim series
    cossim_series = adata_predicted_z.var["cossim"].dropna()
    all_genes = np.array(adata_z.var_names, dtype=str)

    # If too few genes overall, pick randomly
    if cossim_series.size < 9:
        n_pick = min(9, all_genes.size)
        selected_genes = list(np.random.choice(all_genes, size=n_pick, replace=False))
    else:
        vals = cossim_series.values
        genes = np.array(cossim_series.index, dtype=str)

        # Percentile boundaries
        p15, p40, p60, p85 = np.percentile(vals, [15, 40, 60, 85])

        # Candidates per range
        worst = genes[vals <= p15]  # worst: bottom 15%
        mid = genes[(vals >= p40) & (vals <= p60)]  # mid: 40-60%
        best = genes[vals >= p85]  # best: top 15%

        # Fallback: if a group is empty, pick an index slice based on ranking
        sorted_idx = np.argsort(vals)  # ascending
        n = len(vals)

        def fallback_slice(start_frac, end_frac):
            s = int(np.floor(start_frac * n))
            e = int(np.ceil(end_frac * n))
            if e <= s:
                e = min(s + 1, n)
            return genes[sorted_idx[s:e]]

        if worst.size == 0:
            worst = fallback_slice(0.0, 0.15)
        if mid.size == 0:
            mid = fallback_slice(0.4, 0.6)
        if best.size == 0:
            best = fallback_slice(0.85, 1.0)

        # Draw 3 genes per group (with replacement if candidates < 3)
        k = 3
        selected = []
        for grp in (worst, mid, best):
            candidates = np.array(grp, dtype=str)
            if candidates.size == 0:
                continue
            replace = candidates.size < k
            chosen = list(np.random.choice(candidates, size=k, replace=replace))
            selected.extend(chosen)

        selected_genes = selected

    # Generate plots
    for gene in selected_genes:
        cossim_val = float(adata_predicted_z.var.at[gene, "cossim"])
        output_path = (
            metrics_folder_spatial_per_gene
            / f"{cossim_val:.2f}_{gene}_spatial_distribution.pdf"
        )
        generate_gene_spatial_distribution_plot(
            adata_z, adata_predicted_z, gene_name=gene, output_path=output_path
        )


def plot_delta_map(
    adata_z,
    adata_predicted_z,
    gene_name: str,
    mode: str = "relative",
    thresh: float = 1.0,
    show: bool = True,
):
    """
    Delta map for a gene: visualize pred - obs.

    Modes:
    - "relative" (default): delta_normalized = (pred - obs) / max(|pred|, |obs|, eps)
      -> relative change per spot, values typically in [-1, 1], comparable across spots.
    - "absolute": delta = pred - obs
    - "zscore": standardize the delta across spots: (delta - mean)/std

    Spots with very small signal (max(obs, pred) < thresh) are drawn in light grey
    to avoid noise from near-zero values.

    Parameters:
    - thresh: minimum signal for visible spots (default 1.0). Set to 0 to show all spots.
    """
    # Ensure the gene exists
    if (
        gene_name not in adata_z.var_names
        or gene_name not in adata_predicted_z.var_names
    ):
        raise ValueError(
            f"Gene '{gene_name}' nicht in beiden AnnData-Objekten gefunden."
        )

    # Expression vectors per spot (1D)
    def vec_flat(adata, g):
        X = adata[:, g].X
        return X.toarray().ravel() if hasattr(X, "toarray") else np.asarray(X).ravel()

    vec_z = vec_flat(adata_z, gene_name).astype(float)
    vec_pred = vec_flat(adata_predicted_z, gene_name).astype(float)

    # Coordinates
    if "coords" not in adata_z.obsm or "coords" not in adata_predicted_z.obsm:
        raise ValueError(
            "Räumliche Koordinaten (obsm['coords']) fehlen in einem der AnnData-Objekte."
        )
    coords = np.asarray(adata_predicted_z.obsm["coords"])

    # Check shapes
    if coords.shape[0] != vec_z.shape[0] or coords.shape[0] != vec_pred.shape[0]:
        raise ValueError(
            "Anzahl der Spots stimmt nicht mit der Länge der Expressionsvektoren überein."
        )

    eps = 1e-9
    if mode == "absolute":
        delta = vec_pred - vec_z
        label = "Pred - Obs (absolute)"
    elif mode == "relative":
        denom = np.maximum(np.abs(vec_pred), np.abs(vec_z))
        denom = np.maximum(denom, eps)
        delta = (vec_pred - vec_z) / denom
        label = "Relative change (pred - obs) / max(|pred|,|obs|)"
    elif mode == "zscore":
        delta_raw = vec_pred - vec_z
        mu = np.mean(delta_raw)
        sigma = np.std(delta_raw)
        sigma = sigma if sigma > eps else 1.0
        delta = (delta_raw - mu) / sigma
        label = "Z-scored delta (pred - obs)"
    else:
        raise ValueError(
            f"Unbekannter mode='{mode}'. Erlaubt: 'relative', 'absolute', 'zscore'."
        )

    # Mask for low-signal spots (these are drawn grey)
    mask_signal = np.maximum(np.abs(vec_pred), np.abs(vec_z)) >= thresh

    # Symmetric colormap range around 0
    v = (
        np.nanmax(np.abs(delta[mask_signal]))
        if np.any(mask_signal)
        else np.nanmax(np.abs(delta))
    )
    if not np.isfinite(v) or v == 0:
        v = 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    # Background: all spots in light grey
    ax.scatter(
        coords[:, 0], coords[:, 1], c="lightgrey", s=18, alpha=0.6, edgecolors="none"
    )

    # Draw low-signal spots (optionally smaller/faded)
    if np.any(~mask_signal):
        ax.scatter(
            coords[~mask_signal, 0],
            coords[~mask_signal, 1],
            c="lightgrey",
            s=20,
            alpha=0.5,
            edgecolors="none",
        )

    # Plot only the signal-bearing spots with diverging colormap
    sc = ax.scatter(
        coords[mask_signal, 0],
        coords[mask_signal, 1],
        c=delta[mask_signal],
        cmap="RdBu_r",
        vmin=-v,
        vmax=v,
        s=40,
        edgecolors="k",
        linewidths=0.2,
    )

    ax.set_title(f"Delta ({mode}): {gene_name}")
    ax.set_aspect("equal")
    cbar = plt.colorbar(sc, ax=ax, label=label)
    # supplementary info in the colorbar label
    if mode == "relative":
        cbar.set_label(label + " (range approx. [-1,1])")
    if show:
        plt.show()
    plt.close(fig)


def main(
    sc_path: Path, st_path: Path, result_gep: AnnData, metrics_output_folder: Path
):
    """
    Compute metrics for objective 3 and save results as JSON files / diagrams.

    - Compute cosine similarity (and other distance metrics) per gene and save to json; create boxplots for distribution.
    - Compute cosine similarity (and other distance metrics) per spot and save to json; create boxplots for distribution.
    - Generate spatial distribution plots for some genes (high/medium/low cosine similarity).

    Args:
        sc_path: Full path to sc.h5ad.
        st_path: Full path to st.h5ad.
        result_gep: G x S predicted gene expression AnnData
        metrics_output_folder:

    Returns: None
    """
    logger.info("Compute metrics for o2")

    # Assemble file paths
    metrics_folder_boxplots_per_gene = (
        metrics_output_folder / "o2" / "boxplots_per_gene"
    )
    metrics_cossim_per_gene_json = (
        metrics_output_folder / "o2" / "boxplots_per_gene" / "cossim.json"
    )
    metrics_folder_boxplots_per_spot = (
        metrics_output_folder / "o2" / "boxplots_per_spot"
    )
    metrics_cossim_per_spot_json = (
        metrics_output_folder / "o2" / "boxplots_per_spot" / "cossim.json"
    )
    metrics_folder_spatial_per_gene = metrics_output_folder / "o2" / "spatial_per_gene"
    os.makedirs(metrics_folder_boxplots_per_gene, exist_ok=True)
    os.makedirs(metrics_folder_boxplots_per_spot, exist_ok=True)

    # Force remove directory even if not empty
    shutil.rmtree(metrics_folder_spatial_per_gene, ignore_errors=True)
    os.makedirs(metrics_folder_spatial_per_gene, exist_ok=True)

    # Load data (input ST and predicted)
    adata_z, adata_predicted_z = get_z_real_and_predicted_data_only_shared_genes(
        sc_path, st_path, result_gep
    )

    # Assert that both DataFrames have the same shape of genes and spots
    assert (
        adata_z.shape == adata_predicted_z.shape
    ), "DataFrames haben unterschiedliche Formen."

    # Add spatial locations to AnnData objects
    adata_st_full = load_st_adata(st_path)
    spatial = adata_st_full.obsm["spatial"]
    spot_index = {name: i for i, name in enumerate(adata_st_full.obs_names)}
    coords = np.array([spatial[spot_index[s]] for s in adata_z.obs_names])
    adata_z.obsm["coords"] = coords
    adata_predicted_z.obsm["coords"] = coords

    # Compute and store cossim per gene in adata_predicted_z.var
    compute_metrics_per_gene(
        adata_z, adata_predicted_z, save_cossim_json=metrics_cossim_per_gene_json
    )

    # generate_spatial_distribution_plots_for_some_genes(adata_z, adata_predicted_z, metrics_folder_spatial_per_gene)

    # Generate boxplot per gene
    generate_box_plot_metrics_per_gene(
        adata_predicted_z, output_folder=metrics_folder_boxplots_per_gene
    )

    # Compute and store cossim per spot in adata_predicted_z.var
    compute_metrics_per_spot(
        adata_z, adata_predicted_z, save_cossim_json=metrics_cossim_per_spot_json
    )

    # Generate boxplot per spot
    generate_box_plot_metrics_per_spot(
        adata_predicted_z,
        output_folder=metrics_folder_boxplots_per_spot,
    )
