# surrDAMH/modules/continuation.py
import os
import warnings

import numpy as np

from surrDAMH.modules.tools import ensure_dir


def _last_sample_dir(output_dir: str, stage_name: str) -> str:
    return os.path.join(output_dir, "sampling_output", "last_sample", stage_name)


def _carry_over_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "sampling_output", "carry_over")


def save_last_sample(conf, stage_name: str, rank_world: int, parameters: np.ndarray) -> None:
    directory = ensure_dir(_last_sample_dir(conf.output_dir, stage_name))
    print(f"Saving last sample for stage {stage_name!r} at rank {rank_world} to {directory!r}", flush=True)
    np.savez(
        os.path.join(directory, f"rank{rank_world:04d}.npz"),
        parameters=np.asarray(parameters, dtype=np.float64),
        no_parameters=conf.no_parameters,
    )


def save_carry_over(conf, stage_name: str, carried_stage: dict, summary: dict) -> None:
    """
    Persist one adaptive stage's cross-rank hand-over to ``sampling_output/carry_over/<stage_name>.npz``
    (2026-09-21), so a report can show the proposal the following stage actually started from.

    ``carried_stage`` is ``Proposal.carry_over()`` (what ``build_proposal`` consumes for the
    next stage) and ``summary`` is ``Proposal.adapted_summary()`` (diagnostic-only, never
    consumed). Every rank of an MPI run pools to the SAME state (see
    ``GaussRandomWalk_adaptive.set_pooled_state``), so this is written once, by sampler rank 0
    -- unlike ``save_last_sample``, there is no ``rank%04d`` suffix.

    Every value is stored as a numpy array of its own dtype (a scalar as a 0-d array, so an
    ``int`` such as ``n_pooled`` stays integer and a ``float`` stays float); ``load_carry_over``
    reverses that. This is unconditional, like ``save_last_sample`` -- it does not check
    ``Stage.save_to_file`` (there is currently no separate on/off switch for it).
    """
    directory = ensure_dir(_carry_over_dir(conf.output_dir))
    path = os.path.join(directory, f"{stage_name}.npz")
    print(f"Saving carry-over for stage {stage_name!r} to {path!r}", flush=True)
    payload = {}
    for key, value in carried_stage.items():
        payload[f"carry_over__{key}"] = np.asarray(value)
    for key, value in summary.items():
        payload[f"summary__{key}"] = np.asarray(value)
    np.savez(path, **payload)


def load_carry_over(output_dir: str, stage_name: str) -> tuple[dict, dict] | None:
    """
    Read one stage's carry-over file written by :func:`save_carry_over`, or ``None`` if it
    does not exist (a non-adaptive stage, or a run that predates ``carry_over/``).

    Args:
        output_dir: the run's ``Configuration.output_dir`` (the directory that CONTAINS
            ``sampling_output/``), same convention as ``read_run``.
        stage_name: the stage's ``alg%04d_...`` name.

    Returns:
        ``(carry_over, summary)`` with the ``carry_over__``/``summary__`` key prefixes
        stripped; a 0-d array is converted back to the Python scalar of its dtype (``int`` or
        ``float``, via ``ndarray.item()``), a higher-dimensional array is returned as a numpy
        array. ``None`` if the file is missing.
    """
    path = os.path.join(_carry_over_dir(output_dir), f"{stage_name}.npz")
    if not os.path.isfile(path):
        return None
    carry_over: dict = {}
    summary: dict = {}
    with np.load(path) as loaded:
        for key in loaded.files:
            if key.startswith("carry_over__"):
                target, short = carry_over, key[len("carry_over__"):]
            elif key.startswith("summary__"):
                target, short = summary, key[len("summary__"):]
            else:
                continue  # forward compatibility: ignore a key this reader does not know
            value = loaded[key]
            target[short] = value.item() if value.ndim == 0 else value
    return carry_over, summary


def load_last_samples(experiment_dir: str, no_parameters: int, no_chains: int, stage: int | str = -1) -> np.ndarray:
    last_sample_root = os.path.join(experiment_dir, "sampling_output", "last_sample")
    stage_names = sorted(os.listdir(last_sample_root))
    if not stage_names:
        raise ValueError(f"No saved last samples found in {experiment_dir!r}")
    stage_name = stage_names[stage] if isinstance(stage, int) else stage
    stage_dir = os.path.join(last_sample_root, stage_name)

    files = sorted(f for f in os.listdir(stage_dir) if f.endswith(".npz"))
    if len(files) < no_chains:
        raise ValueError(
            f"{experiment_dir!r} stage {stage_name!r} has {len(files)} saved chains, "
            f"but {no_chains} are required to continue from it."
        )
    if len(files) != no_chains:
        warnings.warn(
            f"{experiment_dir!r} stage {stage_name!r} has {len(files)} saved chains, but "
            f"{no_chains} were requested; using the first {no_chains} (sorted by filename, i.e. "
            f"chains {files[:no_chains]}), the remaining {len(files) - no_chains} saved chain(s) "
            "are ignored.",
            RuntimeWarning,
            stacklevel=2,
        )

    last_samples = np.zeros((no_chains, no_parameters), dtype=np.float64)  # older files stored float32; cast on load
    for i in range(no_chains):
        with np.load(os.path.join(stage_dir, files[i])) as loaded:
            stored_no_parameters = int(loaded["no_parameters"])
            if stored_no_parameters != no_parameters:
                raise ValueError(
                    f"{files[i]!r} has no_parameters={stored_no_parameters}, expected {no_parameters}"
                )
            last_samples[i, :] = loaded["parameters"]
    return last_samples