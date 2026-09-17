#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run manifest: a machine-readable record of everything that affects the posterior
for one sampling run (``library_notes/09_improvement_plan.md`` WS4, §0 principle 4).

This module is MPI-free on purpose (no ``mpi4py`` import): MPI layout numbers are passed
in as a plain dict (``mpi_layout``) by the caller (``SamplingFramework.run()`` on rank 0,
or ``runner_local.run_local()``), so the module can be unit-tested without MPI and reused
by both runners.

This records the *seed architecture the code uses today* (``seed0 = 10*(no_stages*rank+i)``,
see ``process_SAMPLER.py`` / ``runner_local.py``). It does NOT implement the seed
architecture described in WS4 (``np.random.SeedSequence`` per (chain, stage, stream)); that
depends on open author decisions (G4/G5, ``library_notes/08_safe_changes_plan.md`` §G).
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

from surrDAMH.modules.tools import ensure_dir

MANIFEST_VERSION = 1
FORMAT_VERSION = 1  # current on-disk sampling_output layout; WS9 will bump this to 2

_PACKAGE_DISTRIBUTION_NAMES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "mpi4py": "mpi4py",
    "torch": "torch",
    "sklearn": "scikit-learn",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Best-effort JSON-safe conversion, used for configuration/stage fields.

    Scalars pass through; arrays become shape+dtype (plus values if small); anything
    else (distributions, proposals, solvers, ...) becomes its class name.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        info: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if value.size <= 16:
            info["values"] = value.tolist()
        return info
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return type(value).__name__


def _summarize_public_attrs(obj: Any, include_values: bool = True) -> dict[str, Any]:
    """Public scalar/array attributes of ``obj`` (bound methods, lists of objects, etc. skipped)."""
    result: dict[str, Any] = {}
    for key, value in vars(obj).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (bool, int, float, str)):
            result[key] = value
        elif isinstance(value, np.generic):
            result[key] = value.item()
        elif isinstance(value, np.ndarray):
            info: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if include_values and value.size <= 16:
                info["values"] = value.tolist()
            result[key] = info
    return result


def _summarize_distribution(dist: Any) -> dict[str, Any] | None:
    if dist is None:
        return None
    return {"class": type(dist).__name__, "attributes": _summarize_public_attrs(dist)}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for key, distribution_name in _PACKAGE_DISTRIBUTION_NAMES.items():
        try:
            versions[key] = importlib.metadata.version(distribution_name)
        except Exception:
            versions[key] = None
    return versions


def _git_info(cwd: str) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                                text=True, timeout=5)
        if commit.returncode == 0:
            info["commit"] = commit.stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd,
                                capture_output=True, text=True, timeout=5)
        if branch.returncode == 0:
            info["branch"] = branch.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                                text=True, timeout=5)
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except Exception:
        pass
    return info


def _surrdamh_version() -> str | None:
    try:
        return importlib.metadata.version("surrDAMH")
    except Exception:
        return None


def _configuration_dict(conf: Any, use_surrogate_gradients_requested: bool | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for f in dataclasses.fields(conf):
        result[f.name] = _json_safe(getattr(conf, f.name))
    effective = getattr(conf, "use_surrogate_gradients")
    result["use_surrogate_gradients"] = effective
    if use_surrogate_gradients_requested is not None and use_surrogate_gradients_requested != effective:
        result["use_surrogate_gradients_requested"] = use_surrogate_gradients_requested
    # __post_init__-computed fields that also affect the posterior/reproducibility:
    result["continued_samples"] = _json_safe(getattr(conf, "continued_samples", None))
    return result


def _stage_dict(stage: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for f in dataclasses.fields(stage):
        result[f.name] = _json_safe(getattr(stage, f.name))
    return result


def _surrogate_dict(updater: Any, evaluator: Any) -> dict[str, Any] | None:
    if updater is None and evaluator is None:
        return None
    result: dict[str, Any] = {
        "updater_class": type(updater).__name__ if updater is not None else None,
        "evaluator_class": type(evaluator).__name__ if evaluator is not None else None,
        "updater_parameters": _summarize_public_attrs(updater, include_values=False) if updater is not None else {},
    }
    return result


def _solver_dict(solver_spec: Any, solver_instance: Any) -> dict[str, Any] | None:
    if solver_spec is not None:
        return {"kind": "solver_spec", **dataclasses.asdict(solver_spec)}
    if solver_instance is not None:
        return {"kind": "solver_instance", "class": type(solver_instance).__name__}
    return None


def _seeds_dict(conf: Any, stages: list) -> dict[str, Any]:
    initial_sample_type = conf.initial_sample_type
    per_rank = []
    no_stages = len(stages)
    for rank in range(conf.no_samplers):
        for i in range(no_stages):
            seed0 = 10 * (no_stages * rank + i)
            per_rank.append({
                "rank": rank,
                "stage_index": i,
                "seed0": seed0,
                "proposal_seed": seed0 + 1,
                "algorithm_seed": seed0 + 2,
            })
    reproducible = initial_sample_type != "prior"
    seeds: dict[str, Any] = {
        "formula": "seed0 = 10*(no_stages*rank_world + i); proposal_seed = seed0+1; algorithm_seed = seed0+2",
        "lhs_seed": 0 if initial_sample_type == "lhs" else None,
        "initial_sample_type": initial_sample_type,
        "initial_sample_reproducible": reproducible,
        "per_rank": per_rank,
    }
    if not reproducible:
        seeds["unreproducible_reason"] = (
            "initial_sample_type='prior' draws from the global, unseeded numpy RNG "
            "(finding 1.9 / G4 in library_notes)"
        )
    return seeds


def _continued_from(conf: Any) -> dict[str, Any] | None:
    if conf.initial_sample_type != "continued":
        return None
    source_dir = conf.continued_from_dir
    source_manifest = None
    manifest_path = os.path.join(source_dir, "sampling_output", "run_manifest.json") if source_dir else None
    if manifest_path and os.path.isfile(manifest_path):
        try:
            with open(manifest_path) as f:
                source_manifest = json.load(f)
        except Exception:
            source_manifest = None
    return {"dir": source_dir, "source_manifest": source_manifest}


def _unverified_options(conf: Any, use_surrogate_gradients_requested: bool | None) -> list[str]:
    options = []
    if getattr(conf, "state_dependent_approximation", False):
        options.append("state_dependent_approximation=True (unverified for subchain_max_length > 1, finding 1.1)")
    if (use_surrogate_gradients_requested and not conf.use_surrogate_gradients):
        options.append("use_surrogate_gradients was disabled by SamplingFramework")
    return options


def build_run_manifest(conf, stages, prior, likelihood, *, runner: str,
                       solver_spec=None, solver_instance=None,
                       surrogate_updater=None, surrogate_evaluator=None,
                       mpi_layout: dict | None = None,
                       use_surrogate_gradients_requested: bool | None = None) -> dict:
    """Build the JSON-serialisable run manifest (does not write anything to disk)."""
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../surrDAMH

    torch_num_threads = None
    if "torch" in sys.modules:
        try:
            import torch
            torch_num_threads = torch.get_num_threads()
        except Exception:
            torch_num_threads = None

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "format_version": FORMAT_VERSION,
        "surrdamh_version": _surrdamh_version(),
        "runner": runner,
        "created_at": _now_iso(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "git": _git_info(package_dir),
        "configuration": _configuration_dict(conf, use_surrogate_gradients_requested),
        "stages": [_stage_dict(stage) for stage in stages],
        "prior": _summarize_distribution(prior),
        "likelihood": _summarize_distribution(likelihood),
        "surrogate": _surrogate_dict(surrogate_updater, surrogate_evaluator),
        "solver": _solver_dict(solver_spec, solver_instance),
        "seeds": _seeds_dict(conf, stages),
        "mpi": dict(mpi_layout) if mpi_layout is not None else None,
        "environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "torch_num_threads": torch_num_threads,
        },
        "unverified_options": _unverified_options(conf, use_surrogate_gradients_requested),
        "continued_from": _continued_from(conf),
    }
    return manifest


def _manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, "sampling_output", "run_manifest.json")


def write_run_manifest(output_dir: str, manifest: dict) -> str:
    """Write ``manifest`` to ``<output_dir>/sampling_output/run_manifest.json``; return its path."""
    ensure_dir(os.path.join(output_dir, "sampling_output"))
    path = _manifest_path(output_dir)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
    return path


def finalize_run_manifest(output_dir: str, extra: dict | None = None) -> None:
    """Re-read the manifest, add ``finished_at`` and whatever ``extra`` holds, write it back."""
    path = _manifest_path(output_dir)
    with open(path) as f:
        manifest = json.load(f)
    manifest["finished_at"] = _now_iso()
    if extra:
        manifest.update(extra)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)


__all__ = ["build_run_manifest", "write_run_manifest", "finalize_run_manifest",
          "MANIFEST_VERSION", "FORMAT_VERSION"]
