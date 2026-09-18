#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reader for one sampling run's on-disk output (**format v2**, see ``docs/outputs.md``
and ``library_notes/11_output_format_v2_spec.md``).

``read_run(output_dir)`` parses everything under ``<output_dir>/sampling_output/`` into a
single :class:`RunData` object: the run manifest, the per-stage/per-chain ``samples`` and
``raw_data`` tables, ``notes``, ``subchain_stats``, ``last_sample`` and the collector's
``surrogate_quality*.csv``. It is the only place in the library that knows the on-disk
column layout; ``surrDAMH.post_processing.Samples`` is a facade on top of it.

Format v2 in one paragraph: every CSV has a header row; ``samples`` is
``multiplicity, par_0..par_{p-1}, log_posterior``; ``raw_data`` is rectangular,
``state_type, par_0..par_{p-1}, solver_tag, obs_0..obs_{m-1}, obs_approx_0..obs_approx_{m-1},
log_likelihood, log_prior``, with the exact block NaN for ``prerejected`` rows and the
surrogate block NaN wherever no surrogate value exists.

A directory without ``run_manifest.json``, or with a ``format_version`` other than
:data:`surrDAMH.modules.manifest.FORMAT_VERSION`, is refused with
:class:`~surrDAMH.modules.manifest.RunFormatError`. There is no converter and no escape
hatch (author decision 6, 2026-09-17).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from surrDAMH.modules.manifest import (FORMAT_VERSION, RunFormatError,
                                       raw_data_columns, samples_columns)

# every ``sampling_output/<data_name>/`` directory that is organized per stage
STAGE_DATA_NAMES = ("samples", "raw_data", "notes", "subchain_stats", "last_sample")

_STAGE_INDEX_PATTERN = re.compile(r"^alg(\d+)_")


@dataclass
class RunData:
    """
    Everything one sampling run wrote, keyed by stage index (see :func:`read_run`).

    Per-stage lists are all ``len(stage_names)`` long and in stage-index order. A stage
    that wrote nothing for a category carries an empty entry, NOT a missing one:

    * ``samples[i] == []`` for a stage with ``Stage.save_to_file=False``;
    * ``raw_data[i] is None`` when ``Configuration.save_snapshots_to_file`` was off (or
      when ``read_run(..., load_raw_data=False)`` skipped it);
    * ``subchain_stats[i] is None`` for a non-DAMH stage;
    * ``notes[i]`` is an empty ``DataFrame`` for a stage that wrote no notes.
    """

    output_dir: str
    manifest: dict
    no_parameters: int
    no_observations: int
    stage_names: list[str]                      # index order, parsed from the "alg%04d" prefix
    samples_columns: list[str]                  # ["multiplicity", "par_0", ..., "log_posterior"]
    samples: list[list[np.ndarray]]             # [stage][chain] -> (n, 2+p) float64
    notes: list[pd.DataFrame]                   # [stage] -> one row per chain
    subchain_stats: list[pd.DataFrame | None]   # [stage] -> None for non-DAMH stages
    raw_data_columns: list[str]                 # ["state_type", "par_0", ..., "log_prior"]
    raw_data: list[list[pd.DataFrame] | None]   # [stage][chain]; None if not saved/not loaded
    last_sample: dict[str, np.ndarray]          # stage_name -> (no_chains, p) float64
    surrogate_quality: pd.DataFrame | None
    surrogate_quality_test: pd.DataFrame | None
    raw_data_loaded: bool = field(default=True)

    @property
    def no_stages(self) -> int:
        return len(self.stage_names)

    def stage_index(self, stage_name: str) -> int:
        """Index of ``stage_name`` in :attr:`stage_names` (``ValueError`` if unknown)."""
        return self.stage_names.index(stage_name)

    def no_chains(self, stage_index: int) -> int:
        """Number of chains that wrote a ``samples`` file for this stage."""
        return len(self.samples[stage_index])


# ---------------------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------------------
def sampling_output_dir(output_dir: str) -> str:
    return os.path.join(str(output_dir), "sampling_output")


def _rank_files(dirname: str, suffix: str = ".csv") -> list[str]:
    """Sorted full paths of the per-chain files in ``dirname`` (``rank%04d`` sorts by rank)."""
    if not os.path.isdir(dirname):
        return []
    names = sorted(f for f in os.listdir(dirname)
                   if f.endswith(suffix) and os.path.isfile(os.path.join(dirname, f)))
    return [os.path.join(dirname, name) for name in names]


def _read_csv_with_header(path: str, expected_columns: list[str] | None) -> pd.DataFrame:
    """Read one v2 CSV; verify its header against ``expected_columns`` when given."""
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise RunFormatError(
            f"{path}: file is empty, so it has no header row. Output format v2 writes a "
            f"header when the file is created; this file was not written by format v"
            f"{FORMAT_VERSION} (no converter exists, decision 6)."
        ) from exc
    if expected_columns is not None and list(frame.columns) != list(expected_columns):
        raise RunFormatError(
            f"{path}: unexpected CSV header.\n"
            f"  expected (format v{FORMAT_VERSION}): {list(expected_columns)}\n"
            f"  found:                              {list(frame.columns)}\n"
            f"There is no converter for other layouts (decision 6)."
        )
    return frame


def _concat_rank_csvs(dirname: str) -> pd.DataFrame:
    """Concatenate the per-chain CSVs of ``notes``/``subchain_stats`` (read by header name)."""
    frames = [_read_csv_with_header(path, None) for path in _rank_files(dirname)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_manifest(output_dir: str) -> dict:
    path = os.path.join(sampling_output_dir(output_dir), "run_manifest.json")
    if not os.path.isfile(path):
        raise RunFormatError(
            f"{path} not found: this is not a surrDAMH output format v{FORMAT_VERSION} "
            f"directory (a run older than the manifest, or not an output directory at all). "
            f"Pre-v{FORMAT_VERSION} runs cannot be read -- there is no converter "
            f"(author decision 6, 2026-09-17); re-run the sampling to obtain v{FORMAT_VERSION} output."
        )
    with open(path) as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise RunFormatError(f"{path}: expected a JSON object, found {type(manifest).__name__}.")
    found = manifest.get("format_version")
    if found != FORMAT_VERSION:
        raise RunFormatError(
            f"{path}: output format_version {found!r}, expected {FORMAT_VERSION}. "
            f"Runs written in another format cannot be read -- there is no converter "
            f"(author decision 6, 2026-09-17); re-run the sampling to obtain "
            f"v{FORMAT_VERSION} output."
        )
    return manifest


def _stage_names_in_index_order(sampling_dir: str) -> list[str]:
    """
    Stage directory names sorted by the index parsed from their ``alg%04d`` prefix.

    Every ``sampling_output/<data_name>/`` tree is scanned, because a stage with
    ``save_to_file=False`` has no ``samples/`` directory but still has ``last_sample/``.
    The index comes from the name, never from ``sorted(os.listdir(...))``, so a stage that
    is missing (or a directory that was renamed by hand) is reported instead of silently
    shifting every later stage.
    """
    names: set[str] = set()
    for data_name in STAGE_DATA_NAMES:
        dirname = os.path.join(sampling_dir, data_name)
        if not os.path.isdir(dirname):
            continue
        for entry in os.listdir(dirname):
            if os.path.isdir(os.path.join(dirname, entry)):
                names.add(entry)

    indexed: dict[int, str] = {}
    for name in sorted(names):
        match = _STAGE_INDEX_PATTERN.match(name)
        if match is None:
            raise RunFormatError(
                f"{sampling_dir}: stage directory {name!r} does not start with the "
                f"'alg%04d_' stage-index prefix required by output format v{FORMAT_VERSION}."
            )
        index = int(match.group(1))
        if index in indexed:
            raise RunFormatError(
                f"{sampling_dir}: stage index {index} is used by two directories "
                f"({indexed[index]!r} and {name!r})."
            )
        indexed[index] = name
    return [indexed[index] for index in sorted(indexed)]


def _read_last_sample(dirname: str) -> np.ndarray:
    rows = []
    for path in _rank_files(dirname, suffix=".npz"):
        with np.load(path) as loaded:
            rows.append(np.asarray(loaded["parameters"], dtype=float))
    if not rows:
        return np.empty((0, 0))
    return np.vstack(rows)


def _read_optional_csv(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    return _read_csv_with_header(path, None)


# ---------------------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------------------
def read_run(output_dir: str, load_raw_data: bool = True) -> RunData:
    """
    Read one sampling run's output directory (format v2) into a :class:`RunData`.

    Args:
        output_dir: the run's ``Configuration.output_dir`` (the directory that CONTAINS
            ``sampling_output/``).
        load_raw_data: load ``raw_data/`` eagerly. ``False`` leaves ``raw_data`` as
            ``None`` for every stage and is what ``post_processing.Samples`` uses, because
            snapshot files are the largest output of a run and its consumers stream them
            chunk by chunk instead.

    Raises:
        RunFormatError: ``run_manifest.json`` is missing, its ``format_version`` is not
            :data:`~surrDAMH.modules.manifest.FORMAT_VERSION`, a stage directory does not
            carry an ``alg%04d`` index prefix, or a CSV header does not match the v2 layout.
    """
    output_dir = str(output_dir)
    sampling_dir = sampling_output_dir(output_dir)
    manifest = _read_manifest(output_dir)

    configuration: dict[str, Any] = manifest.get("configuration") or {}
    try:
        no_parameters = int(configuration["no_parameters"])
        no_observations = int(configuration["no_observations"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunFormatError(
            f"{sampling_dir}/run_manifest.json: 'configuration.no_parameters' and "
            f"'configuration.no_observations' are required to read format v{FORMAT_VERSION} output."
        ) from exc

    stage_names = _stage_names_in_index_order(sampling_dir)
    expected_samples_columns = samples_columns(no_parameters)
    expected_raw_data_columns = raw_data_columns(no_parameters, no_observations)

    samples: list[list[np.ndarray]] = []
    notes: list[pd.DataFrame] = []
    subchain_stats: list[pd.DataFrame | None] = []
    raw_data: list[list[pd.DataFrame] | None] = []
    last_sample: dict[str, np.ndarray] = {}

    for stage_name in stage_names:
        stage_samples = [
            _read_csv_with_header(path, expected_samples_columns).to_numpy(dtype=float)
            for path in _rank_files(os.path.join(sampling_dir, "samples", stage_name))
        ]
        samples.append(stage_samples)

        notes.append(_concat_rank_csvs(os.path.join(sampling_dir, "notes", stage_name)))

        stats = _concat_rank_csvs(os.path.join(sampling_dir, "subchain_stats", stage_name))
        subchain_stats.append(None if stats.empty else stats)

        raw_dir = os.path.join(sampling_dir, "raw_data", stage_name)
        if not load_raw_data or not os.path.isdir(raw_dir):
            raw_data.append(None)
        else:
            raw_data.append([_read_csv_with_header(path, expected_raw_data_columns)
                             for path in _rank_files(raw_dir)])

        last_sample[stage_name] = _read_last_sample(
            os.path.join(sampling_dir, "last_sample", stage_name))

    return RunData(
        output_dir=output_dir,
        manifest=manifest,
        no_parameters=no_parameters,
        no_observations=no_observations,
        stage_names=stage_names,
        samples_columns=expected_samples_columns,
        samples=samples,
        notes=notes,
        subchain_stats=subchain_stats,
        raw_data_columns=expected_raw_data_columns,
        raw_data=raw_data,
        last_sample=last_sample,
        surrogate_quality=_read_optional_csv(os.path.join(sampling_dir, "surrogate_quality.csv")),
        surrogate_quality_test=_read_optional_csv(
            os.path.join(sampling_dir, "surrogate_quality_test.csv")),
        raw_data_loaded=load_raw_data,
    )


__all__ = ["RunData", "read_run", "RunFormatError", "sampling_output_dir",
           "samples_columns", "raw_data_columns"]
