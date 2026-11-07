"""Import-time side effects must not change the caller's matplotlib state.

Importing :mod:`missingly` (or :mod:`missingly.report`) must leave the
process-wide matplotlib backend exactly as the user configured it.  The check
runs in a fresh interpreter because the test process has already imported
matplotlib and may already have a backend selected.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_PROBE = """
import importlib
import matplotlib
before = matplotlib.get_backend()
importlib.import_module("missingly")
importlib.import_module("missingly.report")
after = matplotlib.get_backend()
print(before, after)
"""


def _run_probe(backend: str) -> tuple[str, str]:
    env = dict(os.environ, MPLBACKEND=backend)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=180,
    )
    before, after = result.stdout.split()
    return before, after


@pytest.mark.parametrize("backend", ["svg", "pdf"])
def test_import_missingly_keeps_user_backend(backend: str) -> None:
    before, after = _run_probe(backend)
    assert before == backend
    assert after == before
