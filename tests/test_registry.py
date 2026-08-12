"""Tests for the reference-aligner registry and its single execution path.

subprocess is monkeypatched, so these run without any aligner env installed.
"""

from pathlib import Path

import pytest

from aim.reference_aligners import registry


def test_reference_aligners_registered_and_consistent():
    assert set(registry.REFERENCE_ALIGNERS) == {"tangram", "tacco", "dot"}
    for name, aligner in registry.REFERENCE_ALIGNERS.items():
        assert aligner.name == name
        assert aligner.conda_env == f"{name}_env"
        assert aligner.script == f"run_{name}.py"
        # Executed by path, so the aligner env needs no spatial-aim install.
        assert aligner.script_path.is_file()
        assert aligner.script_path.parent == Path(registry.__file__).resolve().parent


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


def test_conda_exe_falls_back_to_install_search(monkeypatch):
    # Neither CONDA_EXE nor PATH yields conda (IDE run config / non-init'd
    # shell), but conda IS installed -> the location search finds it.
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        registry, "_search_conda_locations", lambda: "/home/u/miniforge3/bin/conda"
    )
    assert registry.conda_exe() == "/home/u/miniforge3/bin/conda"


def test_conda_exe_resolves_bat_wrapper_to_real_exe(tmp_path, monkeypatch):
    # CONDA_EXE points at condabin\conda.bat (recurses to death under
    # subprocess); conda_exe() must hand back the sibling Scripts\conda.exe.
    root = tmp_path / "miniforge3"
    (root / "condabin").mkdir(parents=True)
    (root / "Scripts").mkdir(parents=True)
    bat = root / "condabin" / "conda.bat"
    bat.write_text("@echo off")
    real = root / "Scripts" / "conda.exe"
    real.write_text("")

    monkeypatch.setenv("CONDA_EXE", str(bat))
    assert registry.conda_exe() == str(real)


def test_prefer_real_exe_passthrough_for_non_bat():
    # A plain executable (typical POSIX conda) is returned unchanged.
    assert registry._prefer_real_exe("/opt/conda/bin/conda") == "/opt/conda/bin/conda"


def test_conda_exe_missing_raises(monkeypatch):
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: None)
    monkeypatch.setattr(registry, "_search_conda_locations", lambda: None)
    with pytest.raises(RuntimeError, match="conda not found"):
        registry.conda_exe()


def test_available_conda_envs_empty_without_conda(monkeypatch):
    # No conda at all (e.g. pip install into a plain venv) -> no reference
    # aligners offered, and no crash.
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setattr(registry.shutil, "which", lambda _name: None)
    monkeypatch.setattr(registry, "_search_conda_locations", lambda: None)
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
    # conda run -n tangram_env python <.../run_tangram.py> ...
    assert cmd[:4] == ["conda", "run", "-n", "tangram_env"]
    # By path, not `python -m`: the aligner env needs no spatial-aim install.
    assert "-m" not in cmd
    assert cmd[4] == "python"
    assert cmd[5] == str(registry.REFERENCE_ALIGNERS["tangram"].script_path)
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
