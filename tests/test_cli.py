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
    assert {"run", "gui", "data"} <= choices
    # `aim map-annotation` was removed: `aim run --start_from_annotation` is the
    # annotation path now.
    assert "map-annotation" not in choices


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


def test_run_agglo_tree_method_choices():
    from aim.aim_config import AGGLO_TREE_METHODS, AIMConfig

    parser = cli._build_parser()
    base = ["run", "--scdata", "a.h5ad", "--stdata", "b.h5ad", "--output_dir", "o"]
    # Default matches AIMConfig's, and every registered linkage is accepted.
    assert parser.parse_args(base).agglo_tree_method == AIMConfig().agglo_tree_method
    for method in AGGLO_TREE_METHODS:
        args = parser.parse_args(base + ["--agglo_tree_method", method])
        assert args.agglo_tree_method == method
    # Anything else is rejected by argparse, not silently passed to the tree.
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(base + ["--agglo_tree_method", "complete"])
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


def test_data_validate_requires_data_root():
    with pytest.raises(SystemExit) as exc:
        cli.main(["data", "validate"])
    assert exc.value.code == 2
