"""Post-mapping analysis package; exposes ``run_analysis``."""

__all__ = ["run_analysis"]


def __getattr__(name: str):
    # Lazy import to avoid an import cycle: analysis.analysis imports metrics,
    # which imports leaf helpers from this package.
    if name == "run_analysis":
        from .analysis import run_analysis

        return run_analysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
