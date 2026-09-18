import os
import torch

from surrDAMH.surrogates.parent import Updater

_UPDATER_REGISTRY: dict[str, type[Updater]] = {}

def register_updater(cls: type[Updater]) -> type[Updater]:
    _UPDATER_REGISTRY[cls.__name__] = cls
    return cls

#: File names of the two files a surrogate restart consists of, inside the state directory
#: (``<experiment_folder>/sampling_output/``). Shared with
#: ``surrDAMH.modules.surrogate_restart.SurrogateRestart``.
SURROGATE_CHECKPOINT_NAME = "surrogate_checkpoint.pt"
SURROGATE_TRAINING_DATA_NAME = "surrogate_training_data.npz"


def surrogate_state_paths(experiment_folder: str) -> tuple[str, str]:
    sampling_output = os.path.join(experiment_folder, "sampling_output")
    checkpoint_path = os.path.join(sampling_output, SURROGATE_CHECKPOINT_NAME)
    data_path = os.path.join(sampling_output, SURROGATE_TRAINING_DATA_NAME)
    return checkpoint_path, data_path


def SurrogateReused(experiment_folder: str, load_optimizer: bool = True, **overrides) -> Updater:
    checkpoint_path, data_path = surrogate_state_paths(experiment_folder)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No surrogate checkpoint found in {experiment_folder!r}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No surrogate training data found in {experiment_folder!r}")

    # weights_only=True (S22): a checkpoint is data, not code -- refuse to unpickle
    # arbitrary objects out of it. Everything stored by the updaters here is tensors,
    # scalars, strings and plain containers, so nothing is lost.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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

