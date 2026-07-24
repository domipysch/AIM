"""Interactive Streamlit GUI for browsing AIM sweep results.

Launched with ``python -m gui --scdata ... --stdata ... --output_dir ...``; the
launcher (``gui/__main__.py``) starts a Streamlit server running ``gui/app.py``.
The GUI drives the AIM sweep in-process (one run root per mapper under the given
output dir) and renders per-K views by reusing the repository's existing plotting
and analysis code -- it does not modify any existing module.
"""

# --- native import-order guard (Windows) ------------------------------------
# On Windows this env intermittently segfaults (access violation) when torch is
# initialised before pandas/pyarrow's Arrow DLLs. ``aim`` imports torch (via
# aim.mapping.base) at package load, so any ``from gui import <submodule>`` that
# reaches ``aim`` before pandas can crash. Importing the Arrow/pandas stack here
# first -- before any submodule pulls in torch -- fixes the load order for every
# entry path. Failures are non-fatal (best-effort ordering only).
try:  # pragma: no cover - environment-dependent
    import pyarrow  # noqa: F401
    import pandas  # noqa: F401
    import anndata  # noqa: F401
except Exception:  # noqa: BLE001
    pass
# ----------------------------------------------------------------------------
