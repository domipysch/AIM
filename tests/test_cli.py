"""Tests for the unified ``aim`` CLI dispatcher.

Cover argument parsing, subcommand wiring, and exit codes only (no
torch/scanpy/squidpy). The heavy handlers (the sweep / aligner / GUI runs) are
covered by the integration tests.
"""

import sys

import pytest

from aim import cli


def test_build_parser_lists_all_subcommands():
    parser = cli._build_parser()
    # The subparsers action holds one choice per registered subcommand.
    subactions = [
        a
        for a in parser._actions
        if hasattr(a, "choices") and a.choices and "run" in a.choices
    ]
    assert subactions, "no subparsers found on the aim parser"
    choices = set(subactions[0].choices)
    assert {"run", "gui", "validate"} <= choices
    # `aim map-annotation` was removed: `aim run --start_from_annotation` is the
    # annotation path now.
    assert "map-annotation" not in choices
    # `aim data validate` became the top-level `aim validate`; the `data` group is gone.
    assert "data" not in choices


def test_building_the_parser_loads_no_heavy_modules():
    # Constructing the CLI must not import the sweep stack; those imports are
    # deferred into the individual command handlers.
    cli._build_parser()
    assert not ({"torch", "scanpy", "squidpy"} & set(sys.modules))


def test_version_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "aim" in capsys.readouterr().out


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_no_subcommand_errors():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_run_without_paths_errors():
    # Neither single-pair nor batch flags provided -> usage error, exit 2.
    with pytest.raises(SystemExit) as exc:
        cli.main(["run"])
    assert exc.value.code == 2


def test_run_linkage_method_choices():
    from aim.aim_config import AIMConfig, LINKAGE_METHODS

    parser = cli._build_parser()
    base = ["run", "--scdata", "a.h5ad", "--stdata", "b.h5ad", "--output_dir", "o"]
    # Default matches AIMConfig's, and every registered linkage is accepted.
    assert parser.parse_args(base).linkage_method == AIMConfig().linkage_method
    for method in LINKAGE_METHODS:
        args = parser.parse_args(base + ["--linkage_method", method])
        assert args.linkage_method == method
    # Anything else is rejected by argparse, not silently passed to the tree.
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(base + ["--linkage_method", "complete"])
    assert exc.value.code == 2


def test_run_start_from_annotation_defaults_to_none():
    from aim.aim_config import AIMConfig

    parser = cli._build_parser()
    base = ["run", "--scdata", "a.h5ad", "--stdata", "b.h5ad", "--output_dir", "o"]
    # Default = Leiden over-clustering, matching AIMConfig's.
    assert parser.parse_args(base).start_from_annotation is None
    assert AIMConfig().start_from_annotation is None
    # Any obs column name is accepted verbatim (validated against the h5ad at run time).
    args = parser.parse_args(base + ["--start_from_annotation", "cellType"])
    assert args.start_from_annotation == "cellType"


def test_run_accepts_every_mapper():
    from aim.aim_config import MAPPING_CHOICES

    parser = cli._build_parser()
    base = ["run", "--scdata", "a.h5ad", "--stdata", "b.h5ad", "--output_dir", "o"]
    # The sweep offers the two in-process mappers and every registered reference
    # aligner.
    for method in MAPPING_CHOICES:
        args = parser.parse_args(base + ["--mapping", method])
        assert args.mapping == method
    # Anything else is rejected by argparse, not passed on to build_mapper.
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(base + ["--mapping", "cell2location"])
    assert exc.value.code == 2


def test_validate_without_paths_errors():
    # Neither single-pair nor pairs.csv mode -> usage error, exit 2.
    with pytest.raises(SystemExit) as exc:
        cli.main(["validate"])
    assert exc.value.code == 2


def test_validate_rejects_mixed_modes():
    with pytest.raises(SystemExit) as exc:
        cli.main(["validate", "--pairs_csv", "p.csv", "--scdata", "a.h5ad"])
    assert exc.value.code == 2
    # --sc_dir/--st_dir belong to the pairs.csv mode only.
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["validate", "--scdata", "a.h5ad", "--stdata", "b.h5ad", "--sc_dir", "d"]
        )
    assert exc.value.code == 2


def test_validate_parses_both_modes():
    from pathlib import Path

    parser = cli._build_parser()
    single = parser.parse_args(["validate", "--scdata", "a.h5ad", "--stdata", "b.h5ad"])
    assert (single.scdata, single.stdata) == (Path("a.h5ad"), Path("b.h5ad"))
    assert single.pairs_csv is None

    batch = parser.parse_args(["validate", "--pairs_csv", "DATA/pairs.csv"])
    assert batch.pairs_csv == Path("DATA/pairs.csv")
    # sc_dir/st_dir default to scRNA/ and ST/ next to the CSV, resolved by the runner.
    assert batch.sc_dir is None and batch.st_dir is None
