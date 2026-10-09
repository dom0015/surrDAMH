#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The selection mask of a report (2026-10-08): which (stage, chain) pairs of a run -- or of a
lineage of continued runs -- the statistics use, and how many leading compressed rows of each
kept chain are dropped as burn-in.

It lives in ``<output_dir>/post_processing_output/selection.json`` of the run the report is
written for (the newest run of a lineage). ``SamplingRun.write_report()`` creates it with the
defaults on first use and applies it on every later call; editing the file and calling
``SamplingRun.load(output_dir).write_report()`` again is the re-run loop. The run manifest is
never modified; the file itself is the record of what was dropped. Layout (``format`` 1)::

    {"format": 1,
     "lineage": ["<dir0>", "<dir1>"],
     "stages": [{"name": "alg0000_MH-adaptive", "output_dir": "<dir0>", "is_excluded": true,
                 "include": [0, 0], "burn_in": [0, 0]}, ...],
     "recommended": null,
     "note": "..."}

``include[c]`` (1 = use chain ``c`` of this stage, 0 = drop it) and ``burn_in[c]`` (leading
compressed rows dropped) are indexed by the chain's ORIGINAL number, the position of its
``rank%04d`` file. ``recommended`` is reserved for suggestions of the (later) verdict
diagnostics and is ``null`` for now.
"""

from __future__ import annotations

import json
import os
from typing import Any

from surrDAMH.modules.run_data import RunData

SELECTION_FORMAT = 1
SELECTION_FILENAME = "selection.json"
SELECTION_NOTE = ("Edit include (1 = use the chain's samples of this stage, 0 = drop) and burn_in (leading "
                  "compressed rows to drop per chain); then re-run the post-processing (see the end of "
                  "report_extended.html). 'recommended' is reserved for suggestions of the verdict "
                  "diagnostics (not implemented yet) and stays null.")


def selection_path(output_dir: str) -> str:
    """``<output_dir>/post_processing_output/selection.json`` (absolute)."""
    return os.path.abspath(os.path.join(str(output_dir), "post_processing_output", SELECTION_FILENAME))


def _is_excluded_by_stage(run_data: RunData) -> dict[str, bool]:
    """``Stage.is_excluded`` of every stage, from the manifests (False where not recorded)."""
    specs = run_data.stage_specs
    flags: dict[str, bool] = {}
    for index, name in enumerate(run_data.stage_names):
        spec = next((s for s in specs if s.get("name") == name), None)
        if spec is None and index < len(specs) and not specs[index].get("name"):
            spec = specs[index]
        flags[name] = bool((spec or {}).get("is_excluded", False))
    return flags


def default_selection(run_data: RunData) -> dict[str, Any]:
    """Every chain of every stage, except the stages marked ``is_excluded`` (burn-in stages)."""
    flags = _is_excluded_by_stage(run_data)
    stages = []
    for index, name in enumerate(run_data.stage_names):
        no_chains = run_data.no_chains(index)
        excluded = flags[name]
        stages.append({"name": name,
                       "output_dir": os.path.abspath(run_data.stage_output_dir[index]
                                                     if run_data.stage_output_dir else run_data.output_dir),
                       "is_excluded": excluded,
                       "include": [0 if excluded else 1] * no_chains,
                       "burn_in": [0] * no_chains})
    return {"format": SELECTION_FORMAT,
            "lineage": [os.path.abspath(d) for d in (run_data.output_dirs or [run_data.output_dir])],
            "stages": stages,
            "recommended": None,
            "note": SELECTION_NOTE}


def write_selection(path: str, selection: dict[str, Any]) -> str:
    """Write ``selection`` as indented JSON (one stage per block); return the path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(selection, f, indent=1)
        f.write("\n")
    return path


def validate_selection(selection: Any, run_data: RunData, source: str) -> dict[str, dict[str, list[int]]]:
    """
    Check ``selection`` against the loaded run(s) and return ``{stage_name: {"include": [...],
    "burn_in": [...]}}`` for every loaded stage. Stages in the selection that are not loaded
    (e.g. earlier runs of a lineage read with ``include_previous=False``) are ignored.

    Raises:
        ValueError: (naming ``source`` and the stage) not a format-1 selection; a loaded stage is
            missing; ``include``/``burn_in`` do not have one entry per chain (rank file) of that
            stage; an ``include`` entry is not 0/1; a ``burn_in`` entry is negative or larger than
            the chain's number of compressed rows.
    """
    if not isinstance(selection, dict) or selection.get("format") != SELECTION_FORMAT:
        raise ValueError(f"{source}: not a selection file of format {SELECTION_FORMAT} "
                         f"(expected a JSON object with \"format\": {SELECTION_FORMAT}).")
    entries = selection.get("stages")
    if not isinstance(entries, list):
        raise ValueError(f"{source}: \"stages\" must be a list of stage entries.")
    by_name = {entry.get("name"): entry for entry in entries if isinstance(entry, dict)}
    validated: dict[str, dict[str, list[int]]] = {}
    for index, name in enumerate(run_data.stage_names):
        entry = by_name.get(name)
        if entry is None:
            raise ValueError(f"{source}: stage {name!r} of the loaded run(s) is missing from the selection "
                             "(delete the file to recreate it with the defaults, or add the stage).")
        no_chains = run_data.no_chains(index)
        include, burn_in = entry.get("include"), entry.get("burn_in", [0] * no_chains)
        for key, values in (("include", include), ("burn_in", burn_in)):
            if not isinstance(values, list) or len(values) != no_chains:
                found = len(values) if isinstance(values, list) else type(values).__name__
                raise ValueError(f"{source}: stage {name!r} has {no_chains} chain(s) (rank files) but its "
                                 f"{key!r} has {found} entr{'y' if found == 1 else 'ies'}.")
        include = [int(v) for v in include]
        burn_in = [int(v) for v in burn_in]
        if any(v not in (0, 1) for v in include):
            raise ValueError(f"{source}: stage {name!r}: include entries must be 0 or 1, got {include}.")
        for chain, rows in enumerate(burn_in):
            no_rows = run_data.samples[index][chain].shape[0]
            if not 0 <= rows <= no_rows:
                raise ValueError(f"{source}: stage {name!r}, chain {chain}: burn_in={rows} but the chain "
                                 f"has {no_rows} compressed row(s).")
        validated[name] = {"include": include, "burn_in": burn_in}
    return validated


def read_selection(path: str, run_data: RunData) -> tuple[dict[str, Any], dict[str, dict[str, list[int]]]]:
    """Read and validate a selection file: ``(the file's content, validate_selection(...))``."""
    with open(path) as f:
        try:
            content = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON ({exc}).") from exc
    return content, validate_selection(content, run_data, path)


__all__ = ["SELECTION_FORMAT", "SELECTION_FILENAME", "SELECTION_NOTE", "selection_path",
           "default_selection", "write_selection", "validate_selection", "read_selection"]
