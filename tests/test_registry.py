"""Tests for the reference-aligner registry and its single execution path.

subprocess is monkeypatched, so these run without any aligner env installed.
"""

import pytest

from aim.reference_aligners import registry


def test_reference_aligners_registered_and_consistent():
    assert set(registry.REFERENCE_ALIGNERS) == {"tangram", "tacco", "dot"}
    for name, aligner in registry.REFERENCE_ALIGNERS.items():
        assert aligner.name == name
        assert aligner.conda_env == f"{name}_env"
        assert aligner.module == f"aim.reference_aligners.run_{name}"


def test_run_aligner_unknown_name_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown reference aligner"):
        registry.run_aligner(
            "does-not-exist",
            tmp_path / "sc.h5ad",
            tmp_path / "st.h5ad",
            tmp_path / "out",
            "cellType",
        )


def test_conda_exe_prefers_env_var(monkeypatch):
    # CONDA_EXE wins even when a conda exists on PATH.
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    monkeypatch.setattr(registry.shutil, "which", lambda _name: "/usr/bin/conda")
    assert registry.conda_exe() == "/opt/conda/bin/conda"


def test_conda_exe_falls_back_to_path(monkeypatch):
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setattr(registry.shutil, "which", lambda name: "/usr/bin/conda")
    assert registry.conda_exe() == "/usr/bin/conda"


def test_conda_exe_missing_raises(monkeypatch):
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="conda not found"):
        registry.conda_exe()


def test_available_conda_envs_empty_without_conda(monkeypatch):
    # No conda at all (e.g. pip install into a plain venv) -> no reference
    # aligners offered, and no crash.
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: None)
    assert registry.available_conda_envs() == set()
    assert registry.available_reference_aligners() == []


def test_run_aligner_builds_conda_run_command(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Simulate the aligner writing its output mapping.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / registry.MAPPING_PROB_FILENAME).write_text("stub")

        class _Completed:
            returncode = 0

        return _Completed()

    monkeypatch.setenv("CONDA_EXE", "conda")
    monkeypatch.setattr(registry.subprocess, "run", fake_run)

    result = registry.run_aligner(
        "tangram",
        tmp_path / "sc.h5ad",
        tmp_path / "st.h5ad",
        out_dir,
        "cellType",
    )

    cmd = captured["cmd"]
    # conda run -n tangram_env python -m aim.reference_aligners.run_tangram ...
    assert cmd[:4] == ["conda", "run", "-n", "tangram_env"]
    assert "aim.reference_aligners.run_tangram" in cmd
    assert "--cell_type_key" in cmd and "cellType" in cmd
    assert result == out_dir / registry.MAPPING_PROB_FILENAME


def test_run_aligner_missing_output_raises(tmp_path, monkeypatch):
    # Aligner exits 0 but writes nothing -> run_aligner raises.
    monkeypatch.setenv("CONDA_EXE", "conda")
    monkeypatch.setattr(
        registry.subprocess, "run", lambda cmd, **kw: type("R", (), {"returncode": 0})()
    )
    with pytest.raises(RuntimeError, match="produced no mapping"):
        registry.run_aligner(
            "tacco",
            tmp_path / "sc.h5ad",
            tmp_path / "st.h5ad",
            tmp_path / "out",
            "cellType",
        )
