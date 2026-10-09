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
``read_lineage(output_dir)`` (2026-10-08) reads a run made by ``SamplingRun.continue_sampling``
together with every run it continues, as one ``RunData`` (stages concatenated, oldest first).

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

from surrDAMH.modules.continuation import load_carry_over
from surrDAMH.modules.manifest import (FORMAT_VERSION, RunFormatError,
                                       raw_data_columns, samples_columns)

# every ``sampling_output/<data_name>/`` directory that is organized per stage
STAGE_DATA_NAMES = ("samples", "raw_data", "notes", "subchain_stats", "adaptive_stats",
                    "last_sample")

_STAGE_INDEX_PATTERN = re.compile(r"^alg(\d+)_")
_RANK_FILE_PATTERN = re.compile(r"^rank(\d+)\.")


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
    * ``adaptive_stats[i] is None`` for a stage whose proposal did not adapt;
    * ``carry_over[i] is None`` for a non-adaptive stage, and for an adaptive one whose
      ``sampling_output/carry_over/<stage>.npz`` is missing (a run that predates it);
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
    adaptive_stats: list[pd.DataFrame | None]   # [stage] -> None for non-adaptive stages
    carry_over: list[tuple[dict, dict] | None]  # [stage] -> (carry_over, summary) or None
    raw_data_columns: list[str]                 # ["state_type", "par_0", ..., "log_prior"]
    raw_data: list[list[pd.DataFrame] | None]   # [stage][chain]; None if not saved/not loaded
    last_sample: dict[str, np.ndarray]          # stage_name -> (no_chains, p) float64
    surrogate_quality: pd.DataFrame | None
    surrogate_quality_test: pd.DataFrame | None
    raw_data_loaded: bool = field(default=True)
    # lineage of continued runs (2026-10-08, read_lineage); for read_run: this run alone
    output_dirs: list[str] = field(default_factory=list)       # oldest first, this run last
    stage_output_dir: list[str] = field(default_factory=list)  # [stage] -> output_dir of its run
    same_problem: bool = field(default=True)    # every link of the lineage sampled the same Problem
    generation: int = field(default=0)          # the newest run's generation (0 = not a continuation)
    manifests: list[dict] = field(default_factory=list)        # one per run in output_dirs
    chain_ranks: list[list[int]] = field(default_factory=list)  # [stage][chain] -> rank of its file

    @property
    def no_stages(self) -> int:
        return len(self.stage_names)

    def stage_index(self, stage_name: str) -> int:
        """Index of ``stage_name`` in :attr:`stage_names` (``ValueError`` if unknown)."""
        return self.stage_names.index(stage_name)

    def no_chains(self, stage_index: int) -> int:
        """Number of chains that wrote a ``samples`` file for this stage."""
        return len(self.samples[stage_index])

    @property
    def stage_specs(self) -> list[dict]:
        """The ``stages`` entries of every run's manifest, concatenated in lineage order."""
        manifests = self.manifests or [self.manifest]
        return [dict(spec) for manifest in manifests for spec in (manifest.get("stages") or [])
                if isinstance(spec, dict)]


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


def _ranks_of(paths: list[str]) -> list[int]:
    """Rank number of each ``rank%04d.*`` file (its position if a name does not parse)."""
    ranks = []
    for position, path in enumerate(paths):
        match = _RANK_FILE_PATTERN.match(os.path.basename(path))
        ranks.append(int(match.group(1)) if match else position)
    return ranks


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
    chain_ranks: list[list[int]] = []
    notes: list[pd.DataFrame] = []
    subchain_stats: list[pd.DataFrame | None] = []
    adaptive_stats: list[pd.DataFrame | None] = []
    carry_over: list[tuple[dict, dict] | None] = []
    raw_data: list[list[pd.DataFrame] | None] = []
    last_sample: dict[str, np.ndarray] = {}

    for stage_name in stage_names:
        sample_files = _rank_files(os.path.join(sampling_dir, "samples", stage_name))
        stage_samples = [
            _read_csv_with_header(path, expected_samples_columns).to_numpy(dtype=float)
            for path in sample_files
        ]
        samples.append(stage_samples)
        chain_ranks.append(_ranks_of(sample_files))

        notes.append(_concat_rank_csvs(os.path.join(sampling_dir, "notes", stage_name)))

        stats = _concat_rank_csvs(os.path.join(sampling_dir, "subchain_stats", stage_name))
        subchain_stats.append(None if stats.empty else stats)

        # written only by a stage whose proposal adapts (2026-09-20); absent for every other
        # stage and for every run produced before adaptive_stats existed -> None, not an error
        adaptation = _concat_rank_csvs(os.path.join(sampling_dir, "adaptive_stats", stage_name))
        adaptive_stats.append(None if adaptation.empty else adaptation)

        # written only by sampler rank 0 / run_local, once per adaptive stage (2026-09-21);
        # None for a non-adaptive stage or a run that predates carry_over/ -- never an error
        carry_over.append(load_carry_over(output_dir, stage_name))

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
        adaptive_stats=adaptive_stats,
        carry_over=carry_over,
        raw_data_columns=expected_raw_data_columns,
        raw_data=raw_data,
        last_sample=last_sample,
        surrogate_quality=_read_optional_csv(os.path.join(sampling_dir, "surrogate_quality.csv")),
        surrogate_quality_test=_read_optional_csv(
            os.path.join(sampling_dir, "surrogate_quality_test.csv")),
        raw_data_loaded=load_raw_data,
        output_dirs=[output_dir],
        stage_output_dir=[output_dir] * len(stage_names),
        same_problem=True,
        generation=int((manifest.get("lineage") or {}).get("generation", 0) or 0),
        manifests=[manifest],
        chain_ranks=chain_ranks,
    )


# ---------------------------------------------------------------------------------------
# lineages of continued runs (2026-10-08)
# ---------------------------------------------------------------------------------------
def _previous_link(manifest: dict) -> tuple[str | None, bool]:
    """
    ``(previous output_dir, same_problem)`` of one run, ``(None, True)`` for a run that is not a
    continuation. ``SamplingRun.continue_sampling`` records the link in ``manifest["lineage"]``;
    the old manual continuation (``Configuration(initial_sample_type="continued",
    continued_from_dir=...)``) only in ``manifest["continued_from"]["dir"]``, without saying
    whether the problem was the same -- such a link counts as ``same_problem=False``.
    """
    entry = manifest.get("lineage")
    if isinstance(entry, dict) and entry.get("continued_from"):
        return str(entry["continued_from"]), bool(entry.get("same_problem", True))
    manual = manifest.get("continued_from")
    if isinstance(manual, dict) and manual.get("dir"):
        return str(manual["dir"]), False
    return None, True


def lineage_links(output_dir: str, strict: bool = True) -> tuple[list[str], bool, bool]:
    """
    Walk the lineage of ``output_dir`` backwards through the run manifests.

    Returns:
        ``(dirs, same_problem, complete)``: the output directories oldest first (this run last),
        whether every link has ``same_problem=True``, and whether the walk reached the first run
        (``False`` only with ``strict=False``, when an earlier directory has no readable
        manifest -- e.g. it was moved -- and the walk stopped there).

    Raises:
        RunFormatError: (``strict=True``) an earlier run of the lineage cannot be read, or the
            links form a cycle.
    """
    current = os.path.abspath(str(output_dir))
    dirs = [current]
    seen = {os.path.realpath(current)}
    same_problem = True
    manifest = _read_manifest(current)
    while True:
        previous, same = _previous_link(manifest)
        if previous is None:
            return list(reversed(dirs)), same_problem, True
        previous = os.path.abspath(previous)
        if os.path.realpath(previous) in seen:
            raise RunFormatError(f"{current}: the lineage links form a cycle at {previous!r}.")
        try:
            manifest = _read_manifest(previous)
        except RunFormatError as exc:
            if strict:
                raise RunFormatError(
                    f"{current} continues {previous!r}, which cannot be read as a run of its lineage: "
                    f"{exc} Read this run alone with include_previous=False.") from exc
            return list(reversed(dirs)), same_problem and same, False
        same_problem = same_problem and same
        current = previous
        dirs.append(current)
        seen.add(os.path.realpath(current))


def read_lineage(output_dir: str, load_raw_data: bool = True) -> RunData:
    """
    Read a run together with every run it continues (``SamplingRun.continue_sampling``,
    recursively through ``manifest["lineage"]["continued_from"]``) into ONE :class:`RunData`.

    Per-stage lists are the concatenation of the runs' lists, oldest run first; the stage names
    are unique across the lineage because a continuation numbers its stages after the earlier
    ones. ``manifest``/``output_dir``/``surrogate_quality*`` are those of the newest run;
    ``output_dirs``, ``stage_output_dir``, ``manifests``, ``same_problem`` and ``generation``
    describe the lineage. A run that is not a continuation gives the same data as
    :func:`read_run` (plus these fields).

    Raises:
        RunFormatError: a run of the lineage cannot be read; the runs disagree on
            ``no_parameters``/``no_observations``; or two runs use the same stage name (a lineage
            made with the old manual continuation, ``initial_sample_type="continued"`` without a
            stage-index offset) -- read the newest run alone with ``read_run`` /
            ``Samples(..., include_previous=False)`` then.
    """
    dirs, same_problem, _ = lineage_links(output_dir, strict=True)
    runs = [read_run(directory, load_raw_data=load_raw_data) for directory in dirs]
    newest = runs[-1]
    for run in runs[:-1]:
        if (run.no_parameters, run.no_observations) != (newest.no_parameters, newest.no_observations):
            raise RunFormatError(
                f"lineage of {newest.output_dir}: {run.output_dir} has no_parameters={run.no_parameters}, "
                f"no_observations={run.no_observations} but {newest.output_dir} has "
                f"no_parameters={newest.no_parameters}, no_observations={newest.no_observations}.")
    owner: dict[str, str] = {}
    for run in runs:
        for name in run.stage_names:
            if name in owner:
                raise RunFormatError(
                    f"lineage of {newest.output_dir}: stage name {name!r} is used by both {owner[name]} and "
                    f"{run.output_dir} (a lineage made with the old manual continuation, without a "
                    f"stage-index offset); read this run alone with include_previous=False.")
            owner[name] = run.output_dir
    last_sample: dict[str, np.ndarray] = {}
    for run in runs:
        last_sample.update(run.last_sample)
    return RunData(
        output_dir=newest.output_dir,
        manifest=newest.manifest,
        no_parameters=newest.no_parameters,
        no_observations=newest.no_observations,
        stage_names=[name for run in runs for name in run.stage_names],
        samples_columns=newest.samples_columns,
        samples=[stage for run in runs for stage in run.samples],
        notes=[stage for run in runs for stage in run.notes],
        subchain_stats=[stage for run in runs for stage in run.subchain_stats],
        adaptive_stats=[stage for run in runs for stage in run.adaptive_stats],
        carry_over=[stage for run in runs for stage in run.carry_over],
        raw_data_columns=newest.raw_data_columns,
        raw_data=[stage for run in runs for stage in run.raw_data],
        last_sample=last_sample,
        surrogate_quality=newest.surrogate_quality,
        surrogate_quality_test=newest.surrogate_quality_test,
        raw_data_loaded=load_raw_data,
        output_dirs=[run.output_dir for run in runs],
        stage_output_dir=[run.output_dir for run in runs for _ in run.stage_names],
        same_problem=same_problem,
        generation=newest.generation,
        manifests=[run.manifest for run in runs],
        chain_ranks=[ranks for run in runs for ranks in run.chain_ranks],
    )


__all__ = ["RunData", "read_run", "read_lineage", "lineage_links", "RunFormatError", "sampling_output_dir",
           "samples_columns", "raw_data_columns"]
