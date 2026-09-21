#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rendering of the *effective* settings of a run (WS5, ``library_notes/09_improvement_plan.md``).

``Configuration.describe()`` and ``Stage.describe()`` are built on the two helpers here and
are printed once on rank 0 by ``SamplingFramework.run()`` and by ``run_local()``. The point
is that every dataclass field is shown with the value that is actually in effect at that
moment -- after ``__post_init__`` corrections (e.g. ``adaptive`` forced off for pCN,
``surrogate_model_updates`` resolved from ``None`` / refused for a gradient-free MH stage)
and after ``SamplingFramework`` may have
disabled ``use_surrogate_gradients``. An ignored setting like the ``adaptive_target_rate``
of finding G1 is then visible in the log instead of silently doing nothing.

Fields are wrapped into a few dense lines rather than one line each, so a three-stage run
stays well under 40 lines of output.
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import fields
from typing import Iterable, Sequence

import numpy as np

LINE_WIDTH = 100
INDENT = "  "


def short_repr(value, max_len: int = 44) -> str:
    """
    Compact one-token rendering of a field value for :func:`describe_fields`.

    Scalars and short containers keep their ``repr``; arrays become
    ``ndarray(shape=...)``, anything else its class name in angle brackets, and
    ``sys.maxsize`` (the "no limit" default of ``Stage.max_samples``/``max_evaluations``)
    becomes ``maxsize``.
    """
    if isinstance(value, bool) or value is None:
        text = repr(value)
    elif isinstance(value, int) and value == sys.maxsize:
        text = "maxsize"
    elif isinstance(value, (int, float, str)):
        text = repr(value)
    elif isinstance(value, np.ndarray):
        text = f"ndarray(shape={tuple(value.shape)})"
    elif isinstance(value, (list, tuple, dict, set)):
        text = repr(value)
        if len(text) > max_len:
            text = f"{type(value).__name__}(len={len(value)})"
    else:
        text = f"<{type(value).__name__}>"
    if len(text) > max_len:
        text = text[:max_len - 3] + "..."
    return text


def describe_fields(instance, posterior_affecting: Iterable[str], title: str,
                    extra_lines: Sequence[str] = ()) -> str:
    """
    Render every dataclass field of ``instance`` as ``name=value``, wrapped to a few lines.

    Args:
        instance: a dataclass instance (``Configuration`` or ``Stage``).
        posterior_affecting: field names to mark with a trailing ``*``.
        title: first line of the block.
        extra_lines: derived/effective values that are not dataclass fields (MPI layout,
            requested-vs-effective gradients, ...), appended indented after the fields.

    Returns:
        A multi-line string; every field name of ``instance`` appears in it, which is what
        ``tests/unit/test_describe.py`` pins so a newly added field cannot stay invisible.
    """
    marked = set(posterior_affecting)
    items = [f"{field.name}{'*' if field.name in marked else ''}={short_repr(getattr(instance, field.name))}"
             for field in fields(instance)]
    body = textwrap.wrap(", ".join(items), width=LINE_WIDTH, initial_indent=INDENT,
                         subsequent_indent=INDENT, break_long_words=False, break_on_hyphens=False)
    return "\n".join([title, *body, *(INDENT + line for line in extra_lines)])
