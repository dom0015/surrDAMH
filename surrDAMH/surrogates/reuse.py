import os
import torch

from surrDAMH.surrogates.parent import Updater

_UPDATER_REGISTRY: dict[str, type[Updater]] = {}

def register_updater(cls: type[Updater]) -> type[Updater]:
    _UPDATER_REGISTRY[cls.__name__] = cls
    return cls

def surrogate_state_paths(experiment_folder: str) -> tuple[str, str]:
    sampling_output = os.path.join(experiment_folder, "sampling_output")
    checkpoint_path = os.path.join(sampling_output, "surrogate_checkpoint.pt")
    data_path = os.path.join(sampling_output, "surrogate_training_data.npz")
    return checkpoint_path, data_path


def SurrogateReused(experiment_folder: str, load_optimizer: bool = True, **overrides) -> Updater:
    checkpoint_path, data_path = surrogate_state_paths(experiment_folder)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No surrogate checkpoint found in {experiment_folder!r}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No surrogate training data found in {experiment_folder!r}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    surrogate_type = checkpoint.get("surrogate_type")
    updater_cls = _UPDATER_REGISTRY.get(surrogate_type)
    if updater_cls is None:
        raise ValueError(
            f"Checkpoint surrogate_type {surrogate_type!r} is not registered; "
            f"known types: {sorted(_UPDATER_REGISTRY)}"
        )

    hparams = {**checkpoint.get("updater_hparams", {}), **overrides}
    updater = updater_cls(**hparams)
    updater.load_state(checkpoint_path, data_path, load_optimizer=load_optimizer)
    print(f"Reused surrogate state ({surrogate_type}) from {experiment_folder!r}, "
          f"{updater.no_snapshots} snapshots.", flush=True)
    return updater

