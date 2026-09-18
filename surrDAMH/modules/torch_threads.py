#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic torch intra-op CPU thread count (``Configuration.torch_threads``, author
decision 2026-09-18; see ``library_notes/10_manual_review_notes.md`` §7b "Torch threads").

Background: measured under this container's MPICH ``mpiexec``, every rank already starts
with ``torch.get_num_threads() == 1`` -- but that is a launcher artefact (OpenMP/MKL max
threads throttled to 1 by the launch environment), not something the library itself sets.
Under a different launcher (Slurm ``srun``, OpenMPI, a bare ``python`` process) torch
defaults to *all* visible cores per process, which oversubscribes the node as soon as more
than one rank evaluates/trains the NN surrogate concurrently. This module makes the thread
count explicit and independent of the launcher.

Two mechanisms, chosen by whether torch is already imported when :func:`apply_torch_threads`
runs (must never *import* torch itself, so that a run without an NN surrogate does not pay
for it):

- torch already in ``sys.modules`` -> call ``torch.set_num_threads`` directly. This is the
  path actually taken in this codebase: ``surrDAMH/surrogates/__init__.py`` unconditionally
  imports ``torch_perceptron_minibatches`` (hence torch) as soon as anything does
  ``import surrDAMH`` -- so by the time ``SamplingFramework.run()``/``run_local()`` call this
  helper, torch is *always* already loaded, on every rank, in every launch mode (measured:
  ``import surrDAMH`` alone puts ``"torch"`` in ``sys.modules``). The env-var mechanism below
  is therefore not required for correctness today, but is kept as documented defence in depth
  in case that eager import is ever made lazy.
- torch not yet imported -> best-effort ``os.environ.setdefault`` of ``OMP_NUM_THREADS``/
  ``MKL_NUM_THREADS`` (read by torch/OpenMP at *their* import time, not touched if already
  set by the launcher/user). This is genuinely best-effort: if the OpenMP runtime has
  already been initialized by something else (this process's own numpy import does NOT do
  that -- numpy's BLAS threads are configured lazily, not at import, so plain ``import
  numpy; import surrDAMH`` does not pre-empt this), the env vars are read too late and have
  no effect. The desired value is therefore also stashed here and re-applied the moment
  torch actually gets imported, from ``torch_perceptron_minibatches.py``'s
  ``NeuralNetworkUpdaterMinibatches.__init__``, ``PyTorchNNEvaluator.__init__`` and
  ``PyTorchNNEvaluator.__setstate__`` (the latter is the unpickle hook: an evaluator sent from
  the collector to a sampler over MPI is unpickled via ``__new__``/``__setstate__``, which
  never calls ``__init__``).
"""

from __future__ import annotations

import os
import sys

#: The last value passed to :func:`apply_torch_threads` (``None`` = leave torch's default).
#: Read back by :func:`apply_desired_torch_threads_lazy` so a torch import that happens
#: *after* :func:`apply_torch_threads` (e.g. while unpickling a surrogate evaluator) still
#: ends up with the right thread count.
_desired_torch_threads: int | None = None


def apply_torch_threads(conf) -> None:
    """
    Apply ``conf.torch_threads`` on this rank, once, as early as possible in
    ``SamplingFramework.run()``/``run_local()`` -- before anything evaluates or trains a
    network and before the run manifest is built.

    ``conf.torch_threads is None`` leaves torch's default alone (no-op). Otherwise: if torch
    is already imported, ``torch.set_num_threads`` is called directly; if not, this only sets
    ``OMP_NUM_THREADS``/``MKL_NUM_THREADS`` (via ``setdefault``, never overriding a value the
    launcher/user already set) and records the desired count for
    :func:`apply_desired_torch_threads_lazy` to pick up once torch is actually imported. Never
    imports torch itself.
    """
    global _desired_torch_threads
    n = conf.torch_threads
    _desired_torch_threads = n
    if n is None:
        return
    if "torch" in sys.modules:
        import torch  # already imported elsewhere; this is a name lookup, not a fresh import
        torch.set_num_threads(n)
    else:
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        os.environ.setdefault("MKL_NUM_THREADS", str(n))


def apply_desired_torch_threads_lazy() -> None:
    """
    Re-apply the thread count recorded by the most recent :func:`apply_torch_threads` call,
    if any. Call this from a torch-using constructor/unpickle hook, at a point where torch is
    guaranteed to already be imported (this module never imports torch itself) -- it makes the
    effective thread count correct regardless of whether torch happened to be imported before
    or after :func:`apply_torch_threads` ran on this rank.

    No-op if :func:`apply_torch_threads` was never called (``_desired_torch_threads is None``,
    the process-wide default before any ``Configuration`` was constructed) or if it was called
    with ``torch_threads=None`` (leave torch's default alone).

    Known limitation of this process-wide (not per-``Configuration``) state: an
    ``Updater``/``Evaluator`` is always constructed by the user's script *before*
    ``SamplingFramework.run()``/``run_local()`` calls :func:`apply_torch_threads` for that
    run, so its constructor sees whatever ``_desired_torch_threads`` a *previous* run in the
    same process left behind (``None`` in a fresh process -- the common case, one run per
    process/``mpiexec`` launch). For a concrete (non-``None``) ``torch_threads`` this is only
    transient: the current run's own :func:`apply_torch_threads` call runs moments later and
    (torch always being already imported in this codebase, see the module docstring) directly
    overwrites it, so the final count is always correct for the current run. For
    ``torch_threads=None`` there is no such later correction (``None`` means "touch nothing"),
    so a *second* sampling run in the same process that opts out with ``torch_threads=None``
    can still end up with the thread count a stale earlier run's value forced on the freshly
    constructed updater/evaluator, rather than whatever the process had before that
    construction. Not a concern for the one-run-per-process launch modes this library
    supports (a fresh ``mpiexec``/``python`` process, or one ``run_local()`` call per script);
    a script that runs several sampling runs back to back in one process and wants
    ``torch_threads=None`` to mean "untouched" for every one of them should call
    ``torch.set_num_threads`` itself right before each such run instead of relying on this
    module's memory across runs.
    """
    if _desired_torch_threads is None:
        return
    import torch
    if torch.get_num_threads() != _desired_torch_threads:
        torch.set_num_threads(_desired_torch_threads)
