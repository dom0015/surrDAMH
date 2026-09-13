# surrDAMH/modules/continuation.py
import os
import numpy as np

from surrDAMH.modules.tools import ensure_dir


def _last_sample_dir(output_dir: str, stage_name: str) -> str:
    return os.path.join(output_dir, "sampling_output", "last_sample", stage_name)


def save_last_sample(conf, stage_name: str, rank_world: int, parameters: np.ndarray) -> None:
    directory = ensure_dir(_last_sample_dir(conf.output_dir, stage_name))
    print(f"Saving last sample for stage {stage_name!r} at rank {rank_world} to {directory!r}", flush=True)
    np.savez(
        os.path.join(directory, f"rank{rank_world:04d}.npz"),
        parameters=np.asarray(parameters, dtype=np.float32),
        no_parameters=conf.no_parameters,
    )


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

    last_samples = np.zeros((no_chains, no_parameters), dtype=np.float32)
    for i in range(no_chains):
        with np.load(os.path.join(stage_dir, files[i])) as loaded:
            stored_no_parameters = int(loaded["no_parameters"])
            if stored_no_parameters != no_parameters:
                raise ValueError(
                    f"{files[i]!r} has no_parameters={stored_no_parameters}, expected {no_parameters}"
                )
            last_samples[i, :] = loaded["parameters"]
    return last_samples