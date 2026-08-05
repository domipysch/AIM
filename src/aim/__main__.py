"""Enable ``python -m aim ...`` as an alias for the ``aim`` console script."""

from aim.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
