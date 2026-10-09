#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The selection mask of a report (2026-10-08, layout simplified 2026-10-09): which (stage, chain)
pairs of a run -- or of a lineage of continued runs -- form the posterior (the pooled statistics),
and, optionally, how many leading compressed rows of each included chain are dropped.

It lives in ``<output_dir>/post_processing_output/selection.json`` of the run the report is
written for (the newest run of a lineage). ``SamplingRun.write_report()`` creates it with the
defaults on first use and applies it on every later call; editing the file and calling
``SamplingRun.load(output_dir).write_report()`` again is the re-run loop. The run manifest is
never modified; the file itself is the record of what was dropped. Layout (2026-10-09)::

    {"posterior": {"alg0000_MH-adaptive": [0, 0], "alg0001_DAMH-SMU": [1, 1]},
     "runs": ["<dir0>", "<dir1>"],
     "drop_first_rows": {"alg0001_DAMH-SMU": [5, 0]},
     "note": "..."}

``posterior[stage][c]`` (1 = chain ``c`` of that stage is part of the posterior, 0 = it is not) is
indexed by the chain's ORIGINAL number, the position of its ``rank%04d`` file; one key per stage of
the loaded run(s), in stage order. ``runs`` is written only when the lineage has more than one run
(informational: the absolute output directory of every run it spans; the loader does not read it).
``drop_first_rows[stage][c]`` (leading compressed rows left out of the posterior, an expert entry)
is OPTIONAL and NEVER written by default; a stage/chain absent from it drops 0 rows. There is no
``format``/``lineage``/``stages``/``is_excluded``/``burn_in_stage``/``recommended`` any more (that
layout, used 2026-10-08/09, is rejected with a message to delete the file).

The validated mask returned by :func:`validate_selection`/:func:`read_selection` keeps its
pre-2026-10-09 shape, ``{stage_name: {"include": [...], "burn_in": [...]}}`` (every consumer in
``post_processing/loading.py`` reads only that shape, unaffected by this file's layout).

Since 2026-10-09 the mask selects the posterior only: every chain of every stage is loaded and
shown in the report, the pooled sections use ``Samples.posterior_view()``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from surrDAMH.modules.run_data import RunData

SELECTION_FILENAME = "selection.json"

_OLD_LAYOUT_MESSAGE = ("the selection file layout changed on 2026-10-09 (the old \"format\"/"
                       "\"lineage\"/\"stages\"/\"is_excluded\"/\"burn_in_stage\"/\"recommended\" keys "
                       "are gone, replaced by \"posterior\"/\"runs\"/\"drop_first_rows\"); delete "
                       "this file so it is re-created with the defaults.")


def selection_path(output_dir: str) -> str:
    """``<output_dir>/post_processing_output/selection.json`` (absolute)."""
    return os.path.abspath(os.path.join(str(output_dir), "post_processing_output", SELECTION_FILENAME))


def stage_flags(run_data: RunData) -> dict[str, dict[str, bool]]:
    """``{"is_excluded": ..., "burn_in": ...}`` (``Stage`` flags) of every stage, from the manifests
    (False where not recorded)."""
    specs = run_data.stage_specs
    flags: dict[str, dict[str, bool]] = {}
    for index, name in enumerate(run_data.stage_names):
        spec = next((s for s in specs if s.get("name") == name), None)
        if spec is None and index < len(specs) and not specs[index].get("name"):
            spec = specs[index]
        spec = spec or {}
        flags[name] = {"is_excluded": bool(spec.get("is_excluded", False)),
                       "burn_in": bool(spec.get("burn_in", False))}
    return flags


def selection_note(output_dir: str) -> str:
    """The file's ``"note"`` entry: what 0/1 means, ``drop_first_rows`` in one sentence, and the
    exact re-run command for this run's ABSOLUTE ``output_dir``."""
    command = ('python -c \'import surrDAMH; surrDAMH.SamplingRun.load("'
              f'{os.path.abspath(output_dir)}").write_report()\'')
    return ("1 = chain used in the posterior statistics, 0 = not (one entry per chain, in rank "
            "order, per stage, under \"posterior\"). Optional expert entry \"drop_first_rows\": "
            "{stage: [rows per chain]} drops each listed chain's leading compressed rows (absent by "
            f"default). After editing, re-run: {command}")


def default_selection(run_data: RunData) -> dict[str, Any]:
    """Every chain of every stage, except the ``is_excluded`` and ``burn_in`` stages (posterior 0);
    no ``drop_first_rows``; ``runs`` only for a lineage of more than one run."""
    flags = stage_flags(run_data)
    posterior: dict[str, list[int]] = {}
    for index, name in enumerate(run_data.stage_names):
        no_chains = run_data.no_chains(index)
        excluded = flags[name]["is_excluded"] or flags[name]["burn_in"]
        posterior[name] = [0 if excluded else 1] * no_chains
    selection: dict[str, Any] = {"posterior": posterior}
    dirs = [os.path.abspath(d) for d in (run_data.output_dirs or [run_data.output_dir])]
    if len(dirs) > 1:
        selection["runs"] = dirs
    selection["note"] = selection_note(run_data.output_dir)
    return selection


def write_selection(path: str, selection: dict[str, Any]) -> str:
    """Write ``selection`` to ``path`` (2026-10-09 layout): ``indent=1``, but with each stage's
    chain list kept on one line -- plain ``json.dump(..., indent=1)`` would otherwise put one list
    element per line. Return the path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write(_dump(selection))
    return path


def _dump(selection: dict[str, Any]) -> str:
    """A small custom dump: 1-space indent, one ``{stage: [chain values]}`` line per stage."""
    order = [key for key in ("posterior", "runs", "drop_first_rows", "note") if key in selection]
    order += [key for key in selection if key not in order]  # any unexpected extra key, kept last
    lines = ["{"]
    for i, key in enumerate(order):
        tail = "," if i < len(order) - 1 else ""
        value = selection[key]
        if key in ("posterior", "drop_first_rows") and isinstance(value, dict):
            lines.append(f' {json.dumps(key)}: {{')
            items = list(value.items())
            for j, (name, chain_values) in enumerate(items):
                item_tail = "," if j < len(items) - 1 else ""
                lines.append(f'  {json.dumps(name)}: {json.dumps(chain_values)}{item_tail}')
            lines.append(' }' + tail)
        else:
            lines.append(f' {json.dumps(key)}: {json.dumps(value)}{tail}')
    lines.append("}")
    return "\n".join(lines) + "\n"


def validate_selection(selection: Any, run_data: RunData, source: str) -> dict[str, dict[str, list[int]]]:
    """
    Check ``selection`` against the loaded run(s) and return ``{stage_name: {"include": [...],
    "burn_in": [...]}}`` for every loaded stage (``"burn_in"`` holds ``drop_first_rows``, kept under
    its pre-2026-10-09 key since that is the shape every other module reads). Stages in the
    selection that are not loaded (e.g. earlier runs of a lineage read with
    ``include_previous=False``) are ignored.

    Raises:
        ValueError: (naming ``source`` and the stage) the file still has the pre-2026-10-09
            ``"stages"`` layout; not a selection file (no ``"posterior"`` object); a loaded stage is
            missing from ``"posterior"``; ``posterior``/``drop_first_rows`` do not have one entry per
            chain (rank file) of that stage; a ``posterior`` entry is not 0/1; a
            ``drop_first_rows`` entry is negative or larger than the chain's number of compressed
            rows.
    """
    if isinstance(selection, dict) and "stages" in selection:
        raise ValueError(f"{source}: {_OLD_LAYOUT_MESSAGE}")
    if not isinstance(selection, dict) or not isinstance(selection.get("posterior"), dict):
        raise ValueError(f"{source}: not a selection file (expected a JSON object with a "
                         "\"posterior\" entry mapping each stage name to a list of 0/1, one per chain).")
    posterior = selection["posterior"]
    drop_first_rows = selection.get("drop_first_rows", {})
    if not isinstance(drop_first_rows, dict):
        raise ValueError(f"{source}: \"drop_first_rows\" must be an object mapping stage names to "
                         "lists of row counts, one per chain.")
    validated: dict[str, dict[str, list[int]]] = {}
    for index, name in enumerate(run_data.stage_names):
        if name not in posterior:
            raise ValueError(f"{source}: stage {name!r} of the loaded run(s) is missing from "
                             "\"posterior\" (delete the file to recreate it with the defaults, or "
                             "add the stage).")
        no_chains = run_data.no_chains(index)
        include, rows = posterior[name], drop_first_rows.get(name, [0] * no_chains)
        for key, values in (("posterior", include), ("drop_first_rows", rows)):
            if not isinstance(values, list) or len(values) != no_chains:
                found = len(values) if isinstance(values, list) else type(values).__name__
                raise ValueError(f"{source}: stage {name!r} has {no_chains} chain(s) (rank files) but "
                                 f"{key!r} has {found} entr{'y' if found == 1 else 'ies'}.")
        include = [int(v) for v in include]
        rows = [int(v) for v in rows]
        if any(v not in (0, 1) for v in include):
            raise ValueError(f"{source}: stage {name!r}: posterior entries must be 0 or 1, got {include}.")
        for chain, n in enumerate(rows):
            no_rows = run_data.samples[index][chain].shape[0]
            if not 0 <= n <= no_rows:
                raise ValueError(f"{source}: stage {name!r}, chain {chain}: drop_first_rows={n} but "
                                 f"the chain has {no_rows} compressed row(s).")
        validated[name] = {"include": include, "burn_in": rows}
    return validated


def read_selection(path: str, run_data: RunData) -> tuple[dict[str, Any], dict[str, dict[str, list[int]]]]:
    """Read and validate a selection file: ``(the file's content, validate_selection(...))``."""
    with open(path) as f:
        try:
            content = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON ({exc}).") from exc
    return content, validate_selection(content, run_data, path)


__all__ = ["SELECTION_FILENAME", "selection_path", "stage_flags", "selection_note",
           "default_selection", "write_selection", "validate_selection", "read_selection"]
