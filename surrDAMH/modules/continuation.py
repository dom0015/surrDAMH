# surrDAMH/modules/continuation.py
import os
import warnings

import numpy as np

from surrDAMH.modules.tools import ensure_dir


def _last_sample_dir(output_dir: str, stage_name: str) -> str:
    return os.path.join(output_dir, "sampling_output", "last_sample", stage_name)


def _carry_over_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "sampling_output", "carry_over")

def describe_initial_sample_type(conf) -> str:
    """Wording of ``conf.initial_sample_type`` for the start-up line of a chain (2026-10-09):
    ``"an LHS"``, ``"a prior"``, ``"a user-specified"`` or ``"a continued (from <dir>)"``."""
    kind = getattr(conf, "initial_sample_type", "prior")
    if kind == "lhs":
        return "an LHS"
    if kind == "user_specified":
        return "a user-specified"
    if kind == "continued":
        return f"a continued (from {getattr(conf, 'continued_from_dir', None)!r})"
    return f"a {kind}"



def save_last_sample(conf, stage_name: str, rank_world: int, parameters: np.ndarray) -> None:
    directory = ensure_dir(_last_sample_dir(conf.output_dir, stage_name))
    if getattr(conf, "debug", False):
        print(f"Saving last sample for stage {stage_name!r} at rank {rank_world} to {directory!r}", flush=True)
    np.savez(
        os.path.join(directory, f"rank{rank_world:04d}.npz"),
        parameters=np.asarray(parameters, dtype=np.float64),
        no_parameters=conf.no_parameters,
    )


def initial_posterior_statistics(carried: dict, no_parameters: int):
    """The chain's posterior statistics at the start of a run (C1): the carried
    ``"posterior_statistics"`` vector of a continued run (``_merged_carry_over``), else empty.
    ``n_seen`` from ``"posterior_statistics_n_seen"`` (2026-10-11); a carry-over written before
    that key existed gives ``n_seen = n``."""
    from surrDAMH.modules.posterior_statistics import PosteriorStatistics
    vector = carried.get("posterior_statistics")
    if vector is None:
        return PosteriorStatistics(int(no_parameters))
    n_seen = carried.get("posterior_statistics_n_seen")
    return PosteriorStatistics.from_vector(vector, int(no_parameters),
                                           n_seen=None if n_seen is None else int(np.asarray(n_seen)))


def record_posterior_statistics(stage_carry_over: dict, summary: dict, statistics) -> None:
    """Add the pooled posterior statistics to one stage's carry-over (C1): the vector
    ``as_vector()`` under ``"posterior_statistics"`` (consumed by the next stage / run), the
    undiscounted count ``n_seen`` under ``"posterior_statistics_n_seen"`` (2026-10-11, consumed
    likewise) and the count under ``summary["posterior_statistics_n"]`` (report only)."""
    stage_carry_over["posterior_statistics"] = statistics.as_vector()
    stage_carry_over["posterior_statistics_n_seen"] = int(statistics.n_seen)
    summary["posterior_statistics_n"] = int(statistics.n)


def discount_burn_in_statistics(stage, statistics, summary: dict) -> int | None:
    """
    C1b (2026-10-11): at the end of a ``burn_in=True`` stage, AFTER the cross-rank pooling and the
    adaptive hand-over (so the burn-in stage's own carried proposal is unchanged), reduce the weight
    of the pooled posterior statistics to ``burn_in_statistics_count(d)`` states, keeping their mean
    and covariance (``PosteriorStatistics.discount``), so that the early transient of the warm-up
    no longer dominates the estimate the following stages build on. Recorded as
    ``summary["posterior_statistics_discounted_from_n"]`` (carry-over file). ``is_excluded``
    stages are NOT discounted (their states are not a transient of the next stage: the chain
    restarts) -- and ``burn_in`` and ``is_excluded`` exclude each other anyway; no other stage is.
    Returns the previous count, or ``None`` if nothing was discounted.
    """
    from surrDAMH.modules.posterior_statistics import burn_in_statistics_count
    if not getattr(stage, "burn_in", False):
        return None
    previous = statistics.discount(burn_in_statistics_count(statistics.d))
    if previous is not None:
        summary["posterior_statistics_discounted_from_n"] = int(previous)
    return previous


def save_carry_over(conf, stage_name: str, carried_stage: dict, summary: dict) -> None:
    """
    Persist one stage's cross-rank hand-over to ``sampling_output/carry_over/<stage_name>.npz``
    (2026-09-21), so a report can show the proposal the following stage actually started from.
    Written for EVERY stage since C1 (2026-10-11): ``carry_over__posterior_statistics`` (the pooled
    ``PosteriorStatistics.as_vector()``), ``carry_over__posterior_statistics_n_seen`` (its
    undiscounted count, since 2026-10-11) and ``summary__posterior_statistics_n`` are always present;
    the proposal keys only after an adaptive stage.

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
    if getattr(conf, "debug", False):
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
    does not exist (a run that predates ``carry_over/``; before C1, also a non-adaptive stage).

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
    # TODO: remove this legacy mapping, ignore old format
    # files written before the proposal-spec redesign (2026-09-21) used the old Stage field
    # names as keys; map them so old runs still render in the report
    legacy_keys = {"proposal_sd_or_cov": "scale", "pcn_beta": "beta", "hamiltonian_step_size": "step_size"}
    with np.load(path) as loaded:
        for key in loaded.files:
            if key.startswith("carry_over__"):
                target, short = carry_over, key[len("carry_over__"):]
            elif key.startswith("summary__"):
                target, short = summary, key[len("summary__"):]
            else:
                continue  # forward compatibility: ignore a key this reader does not know
            value = loaded[key]
            if target is carry_over:
                short = legacy_keys.get(short, short)
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