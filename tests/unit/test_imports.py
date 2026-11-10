#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 smoke tests (library_notes/07_testing_plan.md).

WS0 (library_notes/08_safe_changes_plan.md groups A-E) already fixed the import-time
crashes (A1-A3): ``import surrDAMH`` and ``import surrDAMH.process_SAMPLER`` are expected
to succeed cleanly, with no stray output.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.unit
def test_import_surrDAMH_succeeds_with_no_output():
    result = subprocess.run(
        [sys.executable, "-c", "import surrDAMH"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "", f"unexpected stdout on import: {result.stdout!r}"


@pytest.mark.unit
def test_import_process_sampler_succeeds():
    result = subprocess.run(
        [sys.executable, "-c", "import surrDAMH.process_SAMPLER"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
@pytest.mark.skip(
    reason="Sampler-import-without-torch (07_testing_plan.md Phase 0) needs a separate "
    "venv without torch installed; not practical to construct inside this container's "
    "single /dolfinx-env interpreter (all surrogate/torch modules are importable here). "
    "Not tested with a stub either, per the WS1 task brief."
)
def test_import_process_sampler_without_torch():
    ...
