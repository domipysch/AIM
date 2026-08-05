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


@_needs_sample
def test_sweep_writes_expected_outputs(tmp_path):
    """Run a small nearest_centroid sweep and check the on-disk layout."""
    from aim.aim_config import AIMConfig
    from aim.cli import run_one_pair

    out = tmp_path / "run"
    run_one_pair(SC, ST, out, AIMConfig(mapping="nearest_centroid", k_min=1, k_max=2))

    mapping_dir = out / "nearest_centroid"
    assert (mapping_dir / "config.yaml").is_file()
    assert (mapping_dir / "leiden_overclustering.h5ad").is_file()
    assert (mapping_dir / "k_comparison.csv").is_file()

    # Per-K output folders (k_NNN); exclude the k_comparison.csv summary file.
    k_dirs = sorted(p for p in mapping_dir.glob("k_[0-9]*") if p.is_dir())
    assert k_dirs, "no per-K output folders were written"
    for k_dir in k_dirs:
        assert (k_dir / "leiden_to_state.csv").is_file()
        assert (k_dir / "spot_to_state_mapping_soft.h5ad").is_file()
        analysis_data = k_dir / "analysis" / "data"
        assert analysis_data.is_dir() and any(analysis_data.iterdir())


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
