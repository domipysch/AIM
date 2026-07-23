"""Post-mapping analysis for the AIM sweep.

Orchestration only — it loads one K's saved sweep outputs (``loading``), calls
the metric computations in the ``metrics`` package, renders plots (``plots``),
and writes the Typst PDF report (``report``). Everything flows through the
in-memory ``adata_sc`` / ``adata_st`` objects using the keys in
``adata_schema``, so each quantity is computed once and read back by name.

    from analysis import run_analysis
    run_analysis(adata_sc, adata_st, run_dir)
"""

__all__ = ["run_analysis"]


def __getattr__(name: str):
    # Lazy so that importing a leaf helper (e.g. ``analysis.utils``, which
    # ``metrics`` depends on for ``to_dense``) does not eagerly pull in
    # ``analysis.analysis`` — that module imports ``metrics``, which would form
    # an import cycle when ``metrics``/``plots`` is imported before ``analysis``.
    if name == "run_analysis":
        from .analysis import run_analysis

        return run_analysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
