"""End-to-end AIM sweep on the bundled sample dataset.

Marked ``slow``: these run the full sweep (scanpy/squidpy) and so only run where
``aim_env`` and the sample data are present — the CI integration job, not the
fast unit run. Heavy imports happen inside the test bodies, so importing this
module during collection stays light.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]
SC = REPO_ROOT / "sample_dataset" / "scRNA" / "sample_sc.h5ad"
ST = REPO_ROOT / "sample_dataset" / "ST" / "sample_st.h5ad"

_needs_sample = pytest.mark.skipif(
    not (SC.exists() and ST.exists()), reason="sample dataset not available"
)


def _assert_run_layout(mapping_dir):
    """The on-disk layout every sweep writes, whatever the start clusters were."""
    assert (mapping_dir / "config.yaml").is_file()
    assert (mapping_dir / "start_clustering.h5ad").is_file()
    assert (mapping_dir / "k_comparison.csv").is_file()

    # Per-K output folders (k_NNN); exclude the k_comparison.csv summary file.
    k_dirs = sorted(p for p in mapping_dir.glob("k_[0-9]*") if p.is_dir())
    assert k_dirs, "no per-K output folders were written"
    for k_dir in k_dirs:
        assert (k_dir / "start_cluster_to_state.csv").is_file()
        assert (k_dir / "spot_to_state_mapping_soft.h5ad").is_file()
        analysis_data = k_dir / "analysis" / "data"
        assert analysis_data.is_dir() and any(analysis_data.iterdir())
    return k_dirs


@_needs_sample
def test_sweep_writes_expected_outputs(tmp_path):
    """Run a small nearest_centroid sweep and check the on-disk layout."""
    from aim.aim_config import AIMConfig
    from aim.cli import run_one_pair

    out = tmp_path / "run"
    run_one_pair(SC, ST, out, AIMConfig(mapping="nearest_centroid", k_min=1, k_max=2))

    _assert_run_layout(out / "nearest_centroid")


@_needs_sample
def test_sweep_from_annotation_uses_the_annotated_types(tmp_path):
    """--start_from_annotation replaces the Leiden over-clustering.

    The start clusters must be exactly the annotated ``cellType`` values -- one per
    type, named after it -- and no all-gene Leiden resolution is recorded.
    """
    import anndata as ad

    from aim.adata_schema import UNS_LEIDEN_RESOLUTION_ALL_GENES
    from aim.aim_config import AIMConfig
    from aim.cli import run_one_pair

    cell_types = sorted(set(ad.read_h5ad(SC).obs["cellType"].astype(str)))

    out = tmp_path / "run"
    run_one_pair(
        SC,
        ST,
        out,
        AIMConfig(
            mapping="nearest_centroid", k_min=1, start_from_annotation="cellType"
        ),
    )

    mapping_dir = out / "nearest_centroid"
    k_dirs = _assert_run_layout(mapping_dir)

    start = ad.read_h5ad(mapping_dir / "start_clustering.h5ad")
    assert sorted(set(start.obs["start_cluster_name"].astype(str))) == cell_types
    assert set(start.obs["start_cluster"]) == set(range(len(cell_types)))
    # K sweeps from the number of annotated types down to 1.
    assert {int(p.name.split("_")[1]) for p in k_dirs} == set(
        range(1, len(cell_types) + 1)
    )

    scaffold = ad.read_h5ad(out / "reference_scaffold.h5ad")
    assert UNS_LEIDEN_RESOLUTION_ALL_GENES not in scaffold.uns


@_needs_sample
def test_run_cli_end_to_end(tmp_path):
    """The `aim run` entry point maps a single pair and writes the K output."""
    out = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aim",
            "run",
            "--scdata",
            str(SC),
            "--stdata",
            str(ST),
            "--output_dir",
            str(out),
            "--mapping",
            "nearest_centroid",
            "--k_min",
            "1",
            "--k_max",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert (
        out / "nearest_centroid" / "k_001" / "spot_to_state_mapping_soft.h5ad"
    ).is_file()
