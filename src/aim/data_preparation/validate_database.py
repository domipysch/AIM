import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse


def _colored(text: str, status: str) -> str:
    codes = {"OK": "\033[92m", "WARN": "\033[93m", "ERROR": "\033[91m"}
    reset = "\033[0m"
    code = codes.get(status, "")
    return f"{code}{text}{reset}"


def is_intable(x) -> bool:
    try:
        if isinstance(x, (int, np.integer)):
            return True
        if isinstance(x, str):
            int(x)
            return True
    except Exception:
        return False
    return False


def to_int_safe(x) -> int:
    if is_intable(x):
        return int(x)
    raise ValueError("not intable")


def _load_X(adata: ad.AnnData) -> np.ndarray:
    X = adata.X
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def _raw_count_sanity(X: np.ndarray, label: str) -> list[str]:
    warns: list[str] = []
    row_sums = X.sum(axis=1)
    if row_sums.mean() > 0:
        cv = row_sums.std() / row_sums.mean()
        if cv < 0.01:
            warns.append(
                f"{label}: X appears library-size normalized (row-sum CV={cv:.4f})"
            )
    if X.max() < 20:
        warns.append(f"{label}: X max={X.max():.3f} — may be log1p-transformed")
    sample = X.flatten()[:1000]
    if (np.abs(sample - np.round(sample)) > 1e-4).any():
        warns.append(f"{label}: X contains non-integer values — expected raw counts")
    n_zero = (X.sum(axis=1) == 0).sum()
    if n_zero > 0:
        warns.append(f"{label}: {n_zero} observation(s) with all-zero counts")
    return warns


def _check_gene_names_uppercase(adata: ad.AnnData, label: str) -> list[str]:
    non_upper = [g for g in adata.var_names if g != g.upper()]
    if non_upper:
        return [
            f"{label}: {len(non_upper)} gene names are not uppercase "
            f"(e.g. {non_upper[:3]}) — run normalize_gene_names.py"
        ]
    return []


def _check_gene_names_unique(adata: ad.AnnData, label: str) -> list[str]:
    import pandas as pd

    counts = pd.Series(adata.var_names.tolist()).value_counts()
    dups = counts[counts > 1]
    if not dups.empty:
        examples = dups.index.tolist()[:3]
        return [
            f"{label}: {len(dups)} duplicate gene name(s) "
            f"(e.g. {examples}) — rename or remove duplicates"
        ]
    return []


def validate_sc(
    name: str, sc_dir: Path, index_row: pd.Series
) -> tuple[list[str], list[str], ad.AnnData | None]:
    errors: list[str] = []
    warns: list[str] = []

    h5ad_path = sc_dir / f"{name}.h5ad"
    if not h5ad_path.exists():
        errors.append(f"sc: missing file {h5ad_path}")
        return errors, warns, None

    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        errors.append(f"sc: cannot read {h5ad_path} ({e})")
        return errors, warns, None

    errors.extend(_check_gene_names_uppercase(adata, "sc"))
    errors.extend(_check_gene_names_unique(adata, "sc"))

    X = _load_X(adata)

    if np.isnan(X).any():
        errors.append("sc: NaN values in X")
    elif (X < 0).any():
        errors.append("sc: negative values in X")

    warns.extend(_raw_count_sanity(X, "sc"))

    # CellTypeKey0/1/2 + NumberCellTypes0/1/2
    for i in range(3):
        key_col = f"CellTypeKey{i}"
        num_col = f"NumberCellTypes{i}"
        key_val = str(index_row.get(key_col, "")).strip()
        if not key_val:
            break
        if key_val not in adata.obs.columns:
            errors.append(f"sc: '{key_val}' (from {key_col}) not found in obs columns")
        else:
            actual_n = adata.obs[key_val].nunique()
            num_val = str(index_row.get(num_col, "")).strip()
            if not num_val:
                warns.append(f"sc: {num_col} not set (expected count for '{key_val}')")
            elif is_intable(num_val) and to_int_safe(num_val) != actual_n:
                warns.append(
                    f"sc: {num_col}={num_val} != actual unique values in '{key_val}'={actual_n}"
                )

    # Count cross-checks
    for idx_col, label, actual in [
        ("CellCount", "cells", adata.n_obs),
        ("GeneCount", "genes", adata.n_vars),
    ]:
        val = index_row.get(idx_col, "")
        if is_intable(val) and to_int_safe(val) != actual:
            warns.append(f"sc: index.csv {idx_col}={val} != actual {label}={actual}")

    return errors, warns, adata


def validate_st(
    name: str, st_dir: Path, index_row: pd.Series
) -> tuple[list[str], list[str], ad.AnnData | None]:
    errors: list[str] = []
    warns: list[str] = []

    h5ad_path = st_dir / f"{name}.h5ad"
    if not h5ad_path.exists():
        errors.append(f"st: missing file {h5ad_path}")
        return errors, warns, None

    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        errors.append(f"st: cannot read {h5ad_path} ({e})")
        return errors, warns, None

    if "spatial" not in adata.obsm:
        errors.append("st: obsm['spatial'] is missing")
    else:
        spatial = adata.obsm["spatial"]
        if spatial.shape != (adata.n_obs, 2):
            errors.append(
                f"st: obsm['spatial'] shape {spatial.shape} != ({adata.n_obs}, 2)"
            )

    errors.extend(_check_gene_names_uppercase(adata, "st"))
    errors.extend(_check_gene_names_unique(adata, "st"))

    X = _load_X(adata)

    if np.isnan(X).any():
        errors.append("st: NaN values in X")
    elif (X < 0).any():
        errors.append("st: negative values in X")

    warns.extend(_raw_count_sanity(X, "st"))

    # Count cross-checks
    for idx_col, label, actual in [
        ("SpotCount", "spots", adata.n_obs),
        ("GeneCount", "genes", adata.n_vars),
    ]:
        val = index_row.get(idx_col, "")
        if is_intable(val) and to_int_safe(val) != actual:
            warns.append(f"st: index.csv {idx_col}={val} != actual {label}={actual}")

    return errors, warns, adata


def validate_pair(
    pair_id: str,
    sc_name: str,
    st_name: str,
    adata_sc: ad.AnnData,
    adata_st: ad.AnnData,
    index_row: pd.Series,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []

    shared = adata_sc.var_names.intersection(adata_st.var_names)
    n_shared = len(shared)

    if n_shared <= 10:
        warns.append(
            f"pair {pair_id} ({sc_name} × {st_name}): "
            f"only {n_shared} shared genes (expected > 10)"
        )

    val = index_row.get("NumberSharedGenes", "")
    if is_intable(val) and to_int_safe(val) != n_shared:
        warns.append(
            f"pair {pair_id}: NumberSharedGenes={val} != actual shared genes={n_shared}"
        )

    if n_shared > 0:
        st_sub = adata_st[:, shared].X
        if issparse(st_sub):
            st_sub = st_sub.toarray()
        n_zero = (st_sub.sum(axis=1) == 0).sum()
        if n_zero > 0:
            warns.append(
                f"pair {pair_id}: {n_zero} ST spot(s) are all-zero after gene intersection — TACCO will fail"
            )

    return errors, warns


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aim data validate",
        description="Validates scRNA and ST datasets listed in index CSVs (h5ad format)",
    )
    parser.add_argument(
        "--data-root",
        "-d",
        type=Path,
        default=Path(r"/01_Datasets"),
        help="Root folder containing scRNA/, ST/, pairs.csv, scRNA/index.csv, ST/index.csv",
    )
    args = parser.parse_args(argv)

    root: Path = args.data_root
    sc_dir = root / "scRNA"
    st_dir = root / "ST"
    pairs_path = root / "pairs.csv"
    sc_index_path = sc_dir / "index.csv"
    st_index_path = st_dir / "index.csv"

    for p in (pairs_path, sc_index_path, st_index_path):
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            sys.exit(2)

    try:
        pairs_df = pd.read_csv(pairs_path, dtype=str).fillna("")
        sc_index = pd.read_csv(sc_index_path, dtype=str).fillna("").set_index("Name")
        st_index = pd.read_csv(st_index_path, dtype=str).fillna("").set_index("Name")
    except Exception as e:
        print(f"ERROR: cannot read index files: {e}", file=sys.stderr)
        sys.exit(2)

    all_errors: dict[str, list[str]] = {}
    all_warns: dict[str, list[str]] = {}

    adata_sc_cache: dict[str, ad.AnnData | None] = {}
    adata_st_cache: dict[str, ad.AnnData | None] = {}

    # --- Validate scRNA datasets ---
    print("=== scRNA datasets ===")
    for name in sc_index.index:
        row = sc_index.loc[name]
        errs, warns, adata = validate_sc(name, sc_dir, row)
        adata_sc_cache[name] = adata
        if errs:
            all_errors[f"sc:{name}"] = errs
        if warns:
            all_warns[f"sc:{name}"] = warns
        status = "OK" if (not errs and not warns) else ("WARN" if not errs else "ERROR")
        print(
            f"  {name}: {_colored(status, status)} (errors={len(errs)}, warns={len(warns)})"
        )

    # --- Validate ST datasets ---
    print("\n=== ST datasets ===")
    for name in st_index.index:
        row = st_index.loc[name]
        errs, warns, adata = validate_st(name, st_dir, row)
        adata_st_cache[name] = adata
        if errs:
            all_errors[f"st:{name}"] = errs
        if warns:
            all_warns[f"st:{name}"] = warns
        status = "OK" if (not errs and not warns) else ("WARN" if not errs else "ERROR")
        print(
            f"  {name}: {_colored(status, status)} (errors={len(errs)}, warns={len(warns)})"
        )

    # --- Validate pairs ---
    print("\n=== Pairs ===")
    for _, row in pairs_df.iterrows():
        pair_id = row.get("PairID", "?").strip()
        sc_name = row.get("scName", "").strip()
        st_name = row.get("stName", "").strip()

        adata_sc = adata_sc_cache.get(sc_name)
        adata_st = adata_st_cache.get(st_name)

        if adata_sc is None:
            errs = [
                f"pair: scRNA '{sc_name}' could not be loaded (see sc errors above)"
            ]
            all_errors[f"pair:{pair_id}"] = errs
            print(
                f"  pair {pair_id} ({sc_name} × {st_name}): {_colored('ERROR', 'ERROR')} (errors=1, warns=0)"
            )
            continue
        if adata_st is None:
            errs = [f"pair: ST '{st_name}' could not be loaded (see st errors above)"]
            all_errors[f"pair:{pair_id}"] = errs
            print(
                f"  pair {pair_id} ({sc_name} × {st_name}): {_colored('ERROR', 'ERROR')} (errors=1, warns=0)"
            )
            continue

        errs, warns = validate_pair(pair_id, sc_name, st_name, adata_sc, adata_st, row)
        if errs:
            all_errors[f"pair:{pair_id}"] = errs
        if warns:
            all_warns[f"pair:{pair_id}"] = warns
        status = "OK" if (not errs and not warns) else ("WARN" if not errs else "ERROR")
        print(
            f"  pair {pair_id} ({sc_name} × {st_name}): {_colored(status, status)} (errors={len(errs)}, warns={len(warns)})"
        )

    # --- Summary ---
    if all_warns:
        print("\nWARNINGS:")
        for key, msgs in all_warns.items():
            print(f"  {key}:")
            for msg in msgs:
                print(f"    - {msg}")

    if all_errors:
        print("\nERRORS:", file=sys.stderr)
        for key, msgs in all_errors.items():
            print(f"  {key}:", file=sys.stderr)
            for msg in msgs:
                print(f"    - {msg}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll datasets and pairs validated successfully (no errors).")
        sys.exit(0)


if __name__ == "__main__":
    main()
