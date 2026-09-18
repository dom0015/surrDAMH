#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Restarting a surrogate model from a previous run (WS5, ``library_notes/09_improvement_plan.md``).

One class, :class:`SurrogateRestart`, replaces the two ``modules.tools`` helpers
(``surrogate_restart_state_has_snapshots`` / ``load_surrogate_restart_state_if_available``)
that every experiment script used to re-implement or re-wire by hand.

Difference to ``surrDAMH.surrogates.reuse.SurrogateReused``: that function *constructs* an
updater from a checkpoint (its hyper-parameters come out of the checkpoint), while this one
restores state into an updater the user has already configured in the sampling script, and
is applied by ``SamplingFramework`` itself on the collector rank.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy.typing as npt

from surrDAMH.surrogates.parent import Updater
from surrDAMH.surrogates.reuse import (SURROGATE_CHECKPOINT_NAME,
                                       SURROGATE_TRAINING_DATA_NAME)

RestartMode = Literal["state", "data", "none"]
RESTART_MODES: tuple[str, ...] = ("state", "data", "none")


class SurrogateRestart:
    """
    Where a run's surrogate state lives, and how much of it to restore at start-up.

    Pass an instance as ``SamplingFramework(..., surrogate_restart=...)``; it is applied
    **on the collector rank only**, just before ``run_COLLECTOR``, because the collector
    rank is the only one that owns an ``Updater``.

    Args:
        state_dir: directory holding ``surrogate_checkpoint.pt`` and
            ``surrogate_training_data.npz`` — i.e. ``<experiment_dir>/sampling_output``
            (the same layout ``surrDAMH.surrogates.reuse.surrogate_state_paths`` produces).
            Point it at a *previous* run's directory to warm-start from it, or at this
            run's own directory to continue where the last run of the same script stopped.
        mode:
            - ``"state"`` (default): ``Updater.load_state`` — network weights, optimizer
              state and the stored snapshots. The updater comes up ``pretrained_ready``, so
              the collector can hand out an evaluator **before the first snapshot arrives**
              and a DAMH/Hamiltonian *first* stage is possible.
            - ``"data"``: ``Updater.load_training_data`` plus one ``train()`` call — the
              snapshots of the old run are reused, but the network starts from scratch.
            - ``"none"``: do nothing (keeps the argument in a script without deleting it).

    Missing files are not an error: a message is printed and the run starts cold
    (``apply`` returns ``None``), which is what a first run of a restartable script does.
    """

    def __init__(self, state_dir: str, mode: RestartMode = "state") -> None:
        if mode not in RESTART_MODES:
            raise ValueError(f"mode must be one of {RESTART_MODES}, got {mode!r}")
        self.state_dir = state_dir
        self.mode: RestartMode = mode

    @property
    def checkpoint_path(self) -> str:
        """``<state_dir>/surrogate_checkpoint.pt``."""
        return os.path.join(self.state_dir, SURROGATE_CHECKPOINT_NAME)

    @property
    def training_data_path(self) -> str:
        """``<state_dir>/surrogate_training_data.npz``."""
        return os.path.join(self.state_dir, SURROGATE_TRAINING_DATA_NAME)

    def describe(self) -> str:
        """One-line description, logged by ``SamplingFramework`` at start-up."""
        return f"SurrogateRestart(mode={self.mode!r}, state_dir={self.state_dir!r})"

    def apply(self, updater: Updater) -> list[npt.NDArray] | None:
        """
        Restore ``updater`` according to ``mode``.

        Returns:
            The restored ``[parameters, observations, multiplicity]`` snapshot arrays, or
            ``None`` if nothing was restored (``mode="none"``, missing files, or an updater
            whose ``load_state``/``load_training_data`` returned nothing). The caller passes
            them to the collector as ``initial_snapshots`` when the updater does not report
            them itself via ``get_initial_snapshots()``.

        Raises:
            Whatever the updater raises: an incompatible checkpoint (``ValueError``) or an
            updater without state persistence (``NotImplementedError``) is a configuration
            error and must not be silently downgraded to a cold start.
        """
        if self.mode == "none":
            return None

        required = ([self.checkpoint_path, self.training_data_path] if self.mode == "state"
                    else [self.training_data_path])
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            print(f"Collector - surrogate restart (mode={self.mode!r}) requested, but "
                  f"{', '.join(missing)} is missing. Starting cold.", flush=True)
            return None

        if self.mode == "state":
            loaded = updater.load_state(checkpoint_path=self.checkpoint_path,
                                        data_path=self.training_data_path,
                                        load_optimizer=True)
            if loaded is None:
                return None
            snapshots = list(loaded)
            print(f"Collector - restored surrogate state from {self.state_dir} "
                  f"({snapshots[0].shape[0]} snapshots, network and optimizer included).", flush=True)
            return snapshots

        loaded = updater.load_training_data(self.training_data_path)
        snapshots = list(loaded)
        updater.train()  # the network itself is NOT restored in this mode, so refit it on the old data
        print(f"Collector - restored surrogate training data only from {self.state_dir} "
              f"({snapshots[0].shape[0]} snapshots); the network was retrained from scratch.", flush=True)
        return snapshots

    def save(self, updater: Updater) -> bool:
        """
        Write ``updater``'s state to ``checkpoint_path``/``training_data_path`` so that a
        later run with the same ``state_dir`` can restore it.

        Call it on the collector rank after ``SamplingFramework.run()`` returns. Note that
        this overwrites the files ``apply()`` reads, which is what "continue where the last
        run stopped" means; use a different ``state_dir`` to keep the input state intact.

        Returns:
            ``True`` if state was written, ``False`` if the updater holds no snapshots
            (nothing is written then, and a message says so).
        """
        if int(getattr(updater, "no_snapshots", 0)) <= 0:
            print(f"Collector - no snapshots collected, surrogate state not saved to {self.state_dir}.", flush=True)
            return False
        os.makedirs(self.state_dir, exist_ok=True)
        updater.save_state(self.checkpoint_path, self.training_data_path)
        print(f"Collector - saved surrogate state to {self.state_dir}.", flush=True)
        return True
