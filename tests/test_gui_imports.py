"""Import-smoke for the GUI submodules.

Catches stale/broken imports in the Streamlit layer (which mypy does not check).
Marked ``slow`` and skipped unless the optional ``gui`` deps are installed, so it
runs in the integration job (aim_env) but is skipped in the fast unit run.
"""

import importlib

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("streamlit", reason="gui extra not installed")

GUI_SUBMODULES = [
    "compute",
    "data_access",
    "render",
    "scaffold",
    "widgets",
]


@pytest.mark.parametrize("submodule", GUI_SUBMODULES)
def test_gui_submodule_imports(submodule):
    importlib.import_module(f"aim.gui.{submodule}")
