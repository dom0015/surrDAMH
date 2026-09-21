#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import sys
import time
import warnings
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from surrDAMH.configuration import Configuration
from surrDAMH.surrogates.parent import Evaluator

TAG_TERMINATE = 0
TAG_READY_TO_RECEIVE = 1
TAG_DATA = 2
TAG_UPDATE = 3  # signal that sampler wants new evaluator if available
TAG_EVALUATOR_OBJECT = 4  # this message contains evaluator object (or None if sampler terminated)
TAG_STOP_UPDATING = 5  # when sampler knows that it will not want new evaluator later
TAG_INITIAL_SURROGATE = 6  # collector -> sampler, once at start-up: can an evaluator be provided?
TAG_FIRST_SNAPSHOT = 10

# Seconds to wait between printing a fatal traceback and calling MPI_Abort. MPI_Abort makes the
# launcher tear the job down immediately, and mpiexec (MPICH/Hydra) then discards whatever its I/O
# forwarding still holds: without this grace period the traceback of a rank that fails while
# another rank is inside MPI_Comm_spawn is lost entirely and the user only sees a silent exit 1
# (measured: 0/3 runs kept the message, 5/5 keep it with 0.5 s). Lives here (not in core.py) so
# the spawned solver children can import it without pulling in the whole framework.
ABORT_GRACE_SECONDS = 0.5

# Idle throttling of the service loops (solvers pool, collector), findings 4.2/M11. Both loops are
# otherwise a pure busy-wait on Iprobe/Get_status and peg a core at 100% even with nothing to do.
# Measured on toy_examples/typical_example.py (4 ranks, MPICH, 32-core container): 4m20s of user
# CPU before, 3m35s after, at unchanged wall time.
#
# The sleep must not be unconditional-per-idle-iteration: with a sub-millisecond forward model the
# pool is "idle" only for the fraction of a millisecond its child needs, and sleeping 1 ms there
# adds that to every evaluation (measured on toy_examples/minimal_example.py, whose solver is an
# analytic formula: 9.4 s -> 13.9 s wall). Hence ServiceLoopThrottle below: spin as before for the
# first IDLE_SPIN_ITERATIONS consecutive idle iterations (cheap Iprobe calls, ~0.1-0.5 ms in total,
# enough to cover a fast solver's turnaround), and only sleep once the loop is idle beyond that.
IDLE_SLEEP_SECONDS = 0.001
IDLE_SPIN_ITERATIONS = 100


class ServiceLoopThrottle:
    """
    Keeps a polling service loop from burning a whole core while it has nothing to do.

    Call :meth:`step` exactly once per loop iteration with whether that iteration did any real
    work. Consecutive idle iterations first spin (no sleep at all, so a fast forward model sees
    the same latency as before this existed) and then sleep ``sleep_seconds`` each.

    Purely a timing device: it sends, receives and reorders nothing, and a loop using it makes the
    same sequence of MPI calls it made before.
    """

    def __init__(self, spin_iterations: int = IDLE_SPIN_ITERATIONS,
                 sleep_seconds: float = IDLE_SLEEP_SECONDS) -> None:
        self.spin_iterations = spin_iterations
        self.sleep_seconds = sleep_seconds
        self.consecutive_idle = 0

    def step(self, did_something: bool) -> bool:
        """Returns True if this call slept (for tests; the loops ignore the return value)."""
        if did_something:
            self.consecutive_idle = 0
            return False
        self.consecutive_idle += 1
        if self.consecutive_idle > self.spin_iterations:
            time.sleep(self.sleep_seconds)
            return True
        return False


# Below this value for MPI_TAG_UB, a run that cannot be bounded in advance (only a time limit) is
# considered at risk. The MPI standard guarantees MPI_TAG_UB >= 32767 only; real implementations
# offer far more (measured here: MPICH 4.3.1 reports 1073741823 = 2^30 - 1), which no realistic
# run exhausts.
TAG_UB_UNBOUNDED_RUN_THRESHOLD = 1 << 20


def check_tag_upper_bound(list_of_stages: List[Any], tag_ub: int | None = None) -> str | None:
    """
    Start-up diagnostic for finding 2.11: warn if this run could plausibly exhaust ``MPI_TAG_UB``.

    Two counters in this module grow by one per full-model evaluation and are used *as MPI tags*:
    ``SolverMPI.tag_solver`` (sampler -> solvers pool request tag) and ``CommSnapshot_sampler.idx``
    (sampler -> collector snapshot tag, starting at ``TAG_FIRST_SNAPSHOT``). Neither wraps. The MPI
    standard only guarantees ``MPI_TAG_UB >= 32767``, so a long run on an implementation at that
    floor would eventually pass an invalid tag.

    Diagnostic only: it never raises and never changes control flow. Called once, on rank 0, from
    ``SamplingFramework.run()``.

    Args:
        list_of_stages: the run's stages; each contributes ``min(max_evaluations, max_samples)``
            to the projection, or "unbounded" if it has neither (a ``time_limit``-only stage).
        tag_ub: the value to check against; ``None`` queries ``MPI.COMM_WORLD`` for it.

    Returns:
        The warning message that was issued, or ``None`` if nothing was warned about (including
        every failure to determine the projection or the limit).
    """
    try:
        if tag_ub is None:
            tag_ub = MPI.COMM_WORLD.Get_attr(MPI.TAG_UB)
        if tag_ub is None:
            return None
        tag_ub = int(tag_ub)

        projected = TAG_FIRST_SNAPSHOT
        has_unbounded_stage = False
        for stage in list_of_stages:
            bound = min(int(getattr(stage, "max_evaluations", sys.maxsize)),
                        int(getattr(stage, "max_samples", sys.maxsize)))
            if bound >= sys.maxsize:
                has_unbounded_stage = True
            else:
                projected += bound

        message = None
        if projected > tag_ub:
            message = (f"MPI_TAG_UB is {tag_ub}, but this run projects up to about {projected} MPI tags "
                       f"(one per full-model evaluation, per sampler, summed over stages). Tags are not "
                       f"reused, so the run may fail with an invalid-tag error before it finishes. "
                       f"Reduce max_evaluations/max_samples or use an MPI implementation with a larger "
                       f"MPI_TAG_UB.")
        elif has_unbounded_stage and tag_ub < TAG_UB_UNBOUNDED_RUN_THRESHOLD:
            message = (f"MPI_TAG_UB is only {tag_ub} and at least one stage has no max_evaluations/"
                       f"max_samples bound (time limit only). MPI tags grow by one per full-model "
                       f"evaluation and are not reused, so a long enough run may fail with an "
                       f"invalid-tag error.")
        if message is not None:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return message
    except Exception:  # diagnostic only - must never be able to break a run
        return None


def _values_equal(own: Any, reference: Any) -> bool:
    """``==`` on two normalized configuration values, never raising and never ambiguous.

    ``Configuration.requested_posterior_fields_for_comparison()`` already reduces everything to
    primitives, but a stray array-like (e.g. from a custom subclass) must not produce
    ``ValueError: truth value of an array ... is ambiguous`` here, so anything whose ``==`` does
    not yield a plain bool falls back to ``np.array_equal``.
    """
    try:
        result = (own == reference)
        if isinstance(result, bool):
            return result
        return bool(np.array_equal(own, reference))
    except Exception:
        return bool(np.array_equal(own, reference))


def check_configuration_consistency(conf: Configuration, comm: Any = None) -> None:
    """
    Fail loudly at start-up if the ranks were given different ``Configuration`` objects (2.9).

    Rank 0 broadcasts the **requested** values (the literal constructor arguments, snapshotted in
    ``Configuration.__post_init__``) of every field in
    ``surrDAMH.configuration.POSTERIOR_AFFECTING_FIELDS``, and every rank compares its own
    snapshot field by field. A script that builds its ``Configuration`` correctly runs the same
    call with the same literal arguments on every rank, so the requested values always match;
    a script that (by bug) builds a different one per rank used to run anyway - sampling a
    different posterior per chain, hanging, or crashing somewhere unrelated later.

    Why the *requested* values and not the effective ones: ``use_surrogate_gradients`` may
    legitimately end up different per rank, since
    ``SamplingFramework._configure_surrogate_gradients()`` disables it locally on a rank whose
    updater/evaluator cannot do gradients (only the collector holds an ``Updater``). Comparing
    the pre-mutation snapshot sidesteps that without needing a per-field exclusion list.

    Not compared: the MPI-layout fields (``no_samplers``, ``rank_collector``,
    ``rank_solvers_pool``, ``sampler_ranks``), which ``__post_init__`` derives from
    ``MPI.COMM_WORLD.Get_size()`` and are identical on every rank by construction; and every
    non-posterior-affecting field (buffer sizes, ``debug``, ...), whose divergence cannot change
    a result.

    Collective: must be called on every rank of ``comm`` (once per role, from
    ``SamplingFramework.run()`` before role dispatch). A single ``bcast`` of a small dict of
    primitives plus one ``allreduce``; the ``allreduce`` makes *every* rank raise when *any* rank
    sees a difference, so the job fails with the same error everywhere instead of one rank
    crashing while the rest block in a matching MPI call.

    Args:
        conf: this rank's configuration.
        comm: communicator to check over; ``None`` means ``MPI.COMM_WORLD``. A size-1
            communicator (or the MPI-free ``run_local()``, which never calls this) has nothing
            to compare against and returns immediately.

    Raises:
        RuntimeError: on every rank, naming the differing field(s) and both values, if any rank's
            requested posterior-affecting fields differ from rank 0's.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        return
    rank = comm.Get_rank()

    local = conf.requested_posterior_fields_for_comparison()
    reference = comm.bcast(local if rank == 0 else None, root=0)

    differences: List[str] = []
    for name in sorted(set(reference) | set(local)):
        if name not in local:
            differences.append(f"{name}: missing on rank {rank}, rank 0 has {reference[name]!r}")
        elif name not in reference:
            differences.append(f"{name}: rank {rank} has {local[name]!r}, missing on rank 0")
        elif not _values_equal(local[name], reference[name]):
            differences.append(f"{name}: rank {rank} has {local[name]!r}, rank 0 has {reference[name]!r}")

    any_mismatch = bool(comm.allreduce(1 if differences else 0, op=MPI.MAX))
    if not any_mismatch:
        return
    if differences:
        raise RuntimeError(
            f"Configuration mismatch across MPI ranks (finding 2.9): rank {rank} was constructed with "
            f"posterior-affecting Configuration field(s) differing from rank 0: " + "; ".join(differences) +
            ". Every rank must construct the same Configuration(...); fix the script so it does not "
            "make any Configuration argument depend on the rank.")
    raise RuntimeError(
        f"Configuration mismatch across MPI ranks (finding 2.9): rank {rank}'s own posterior-affecting "
        "Configuration fields match rank 0's, but at least one other rank reported a difference - see "
        "its traceback for the offending field(s). The whole job is aborted because the ranks would "
        "otherwise sample different posteriors.")


def send_initial_surrogate_availability(rank_sampler: int, can_provide_evaluator: bool) -> None:
    """
    Collector side of the start-up handshake (WS8, finding 2.2).

    Tells one sampler whether the collector can hand out a surrogate evaluator *before*
    receiving any new snapshot, i.e. whether it has a pretrained/restored surrogate or
    already holds at least ``min_snapshots_initial`` snapshots.

    Sent once per sampler, right after the collector's set-up and before its main loop, so
    it never waits for anything the samplers do first. The message has its own tag and is
    consumed exactly once, so it does not interfere with the TAG_UPDATE /
    TAG_EVALUATOR_OBJECT bookkeeping of ``CommEvaluator_collector`` (see its invariant).
    """
    buf = np.array([1 if can_provide_evaluator else 0], dtype=int)
    MPI.COMM_WORLD.Send(buf=buf, dest=rank_sampler, tag=TAG_INITIAL_SURROGATE)


def recv_initial_surrogate_availability(rank_collector: int) -> bool:
    """
    Sampler side of the start-up handshake; see ``send_initial_surrogate_availability``.

    Blocking on purpose: the collector sends this message before its main loop and without
    waiting for the sampler, so it is (or will shortly be) available.
    """
    buf = np.zeros(shape=(1,), dtype=int)
    MPI.COMM_WORLD.Recv(buf=buf, source=rank_collector, tag=TAG_INITIAL_SURROGATE)
    return bool(buf[0])


class CommEvaluator_sampler:
    """
    MPI communication between one sampler and collector regarding surrogate model evaluators,
    sampler side.
    """

    def __init__(self, rank_collector: int, max_buffer_size: int = 1 << 30) -> None:
        self.rank_collector = rank_collector
        self.max_buffer_size = max_buffer_size
        self.comm_world = MPI.COMM_WORLD
        self.idx = 0
        self.evaluator = None
        self.request_Isend = None

    def request_evaluator(self) -> None:
        """
        Sends signal to collector, that he is ready to receive a new evaluator.
        Prepares irecv request for the new evaluator.
        """
        self.idx += 1
        buf = np.array([self.idx], dtype=int)
        if self.request_Isend is not None:
            self.request_Isend.Wait()
        self.request_Isend = self.comm_world.Isend(buf=buf, dest=self.rank_collector, tag=TAG_UPDATE)
        self.request_irecv = self.comm_world.irecv(buf=self.max_buffer_size, source=self.rank_collector, tag=TAG_EVALUATOR_OBJECT)

    def evaluator_is_available(self) -> bool:
        """
        Checks if the surrogate model evaluator can be received without waiting.
        """
        return self.request_irecv.get_status()

    def get_evaluator(self) -> Evaluator:
        """
        Waits for current evaluator.
        """
        evaluator = self.request_irecv.wait()
        self.evaluator = evaluator
        return evaluator

    def get_evaluator_and_terminate(self) -> Evaluator | None:
        """
        Sends termination signal to the collector.
        Receives last evaluator (or None if a new evaluator is not available).
        """
        buf = np.array([self.idx], dtype=int)
        self.comm_world.Send(buf=buf, dest=self.rank_collector, tag=TAG_STOP_UPDATING)
        evaluator = self.request_irecv.wait()
        self.request_irecv = None
        # Complete the last TAG_UPDATE Isend posted by request_evaluator() before dropping the
        # reference (finding 2.10): abandoning a pending non-blocking send is undefined by the MPI
        # standard (the buffer may still be read by the implementation). Same pattern as
        # request_evaluator() itself and as CommEvaluator_collector.terminate(). The message is a
        # single int, so it is sent eagerly and this never blocks; the collector always has a
        # matching TAG_UPDATE Irecv posted (re-posted in sampler_requests_evaluator(), cancelled
        # only in sampler_stops()/terminate()). Wire protocol unchanged.
        if self.request_Isend is not None:
            self.request_Isend.Wait()
        self.request_Isend = None
        self.comm_world = None
        if evaluator is None:  # this means that at least one evaluator has been received before
            return self.evaluator
        return evaluator


class CommEvaluator_collector():
    """
    MPI communication between one sampler and collector regarding surrogate model evaluators,
    collector side.
    On initialization, prepares irecv request for a signal from sampler.

    Invariant: the collector consumes a TAG_UPDATE signal (``sampler_requests_evaluator``)
    only when it simultaneously sends an evaluator (``send_evaluator``); ``terminate()``
    relies on this (``current_idx == max_idx`` <=> every request was answered) to decide
    whether a final TAG_EVALUATOR_OBJECT message is still owed to the sampler.
    """

    def __init__(self, rank_sampler: int) -> None:
        self.rank_sampler = rank_sampler
        self.comm_world = MPI.COMM_WORLD
        self.active = True
        self.current_idx = 0
        self.recv_buffer_update = np.zeros(shape=(1,), dtype=int)
        self.request_update_signal = self.comm_world.Irecv(buf=self.recv_buffer_update, source=self.rank_sampler, tag=TAG_UPDATE)
        self.recv_buffer_stop = np.zeros(shape=(1,), dtype=int)
        self.request_stop_signal = self.comm_world.Irecv(buf=self.recv_buffer_stop, source=self.rank_sampler, tag=TAG_STOP_UPDATING)
        self.isend_request = None

    def sampler_requests_evaluator(self) -> bool:
        """
        Checks if there is an incoming "update" signal from the sampler,
        receives it and prepares a new request.
        """
        if self.request_update_signal.Get_status():
            self.request_update_signal.Wait()
            self.request_update_signal = self.comm_world.Irecv(buf=self.recv_buffer_update, source=self.rank_sampler, tag=TAG_UPDATE)
            self.current_idx = self.recv_buffer_update[0]
            return True
        return False

    def sampler_stops(self) -> bool:
        """
        Checks if there is a "stop updating" signal from the sampler and receives it.
        """
        if self.request_stop_signal.Get_status():
            self.request_update_signal.Cancel()
            self.request_stop_signal.Wait()
            self.max_idx = self.recv_buffer_stop[0]
            self.active = False
            return True
        return False

    def send_evaluator(self, evaluator: Evaluator | None) -> None:
        """
        Sends evaluator to the sampler.
        """
        if self.isend_request is not None:
            self.isend_request.wait()
        self.isend_request = self.comm_world.isend(obj=evaluator, dest=self.rank_sampler, tag=TAG_EVALUATOR_OBJECT)

    def terminate(self, last_evaluator: Evaluator | None) -> None:
        """
        If all update signals have been received,
        cancel last update signal request.
        If the last update signal haven't been received yet,
        send the last evaluator (or None, if this one was sent already),
        wait for the last update signal.
        IF NO EVALUATOR HAVE BEEN SENT YET, MAKE SURE TO DO IT NOW.
        """
        # Ensure any in-flight Isend from send_evaluator() has completed before
        # exiting, to avoid abandoning the MPI request during finalization.
        if self.isend_request is not None:
            self.isend_request.wait()
        if self.current_idx == self.max_idx:
            self.request_update_signal.Cancel()
            self.request_update_signal.Wait()
        else:
            self.comm_world.send(obj=last_evaluator, dest=self.rank_sampler, tag=TAG_EVALUATOR_OBJECT)
            self.request_update_signal.Wait()
        self.request_update_signal = None
        self.request_stop_signal = None
        self.isend_request = None
        self.comm_world = None


class CommSnapshot_sampler:
    """
    MPI communication between one sampler and collector regarding collecting snapshots,
    sampler side.
    Snapshot is a list containing:
        parameters (npt.NDArray)
        observations (npt.NDArray)
        weight (int)
        likelihood (float)
    """

    def __init__(self, rank_collector: int, max_sampler_isend_requests: int = 100) -> None:
        self.rank_collector = rank_collector
        self.max_sampler_isend_requests = max_sampler_isend_requests
        self.comm_world = MPI.COMM_WORLD
        self.requests = []  # buffer for isend requests
        self.idx = TAG_FIRST_SNAPSHOT

    def send_to_collector(self, snapshot: List[Any]):
        """
        Sends snapshot to collector.
        """
        data_to_pickle = copy.deepcopy(snapshot)
        # create isend request for the snapshot
        request_send = self.comm_world.isend(data_to_pickle, dest=self.rank_collector, tag=self.idx)
        # store the isend request in a list
        # if the number of unsent messages is above maximum, wait for one of them
        if (self.idx-TAG_FIRST_SNAPSHOT) >= self.max_sampler_isend_requests:
            idx_sent, _ = MPI.Request.waitany(self.requests)
            self.requests[idx_sent] = request_send
        else:
            self.requests.append(request_send)
        self.idx += 1

    def terminate(self) -> None:
        """
        Sends message to collector, tag indicates that there will be no more snapshots from this sampler,
        message is the tag of the last snapshot sent by this sampler.
        """
        MPI.Request.waitall(self.requests)
        # sends the number of sent snapshots
        buf = np.array([self.idx-1], dtype=int)
        self.comm_world.Send(buf=buf, dest=self.rank_collector, tag=TAG_TERMINATE)
        self.requests = []
        self.comm_world = None


class CommSnapshot_collector:
    """
    MPI communication between one sampler and collector regarding collecting snapshots,
    collector side.
    Snapshot is a list containing:
        parameters (npt.NDArray)
        observations (npt.NDArray)
        weight (int)
        likelihood (float)
    """

    def __init__(self, rank_sampler: int) -> None:
        self.rank_sampler = rank_sampler
        self.comm_world = MPI.COMM_WORLD
        self.sampler_terminated = False  # switch to True after receiving termination signal
        self.all_snapshots_were_received = False  # switch to True after receiving last snapshot
        self.max_idx = np.inf
        # prepare for receiving termination signal
        self.buffer_terminate = np.zeros(shape=(1,), dtype=int)
        self.request_terminate = self.comm_world.Irecv(buf=self.buffer_terminate, source=self.rank_sampler, tag=TAG_TERMINATE)
        # prepare for receiving first snapshot
        self.current_idx = TAG_FIRST_SNAPSHOT
        self.request_snapshot = self.comm_world.irecv(source=self.rank_sampler, tag=self.current_idx)

    def snapshot_is_available(self) -> bool:
        """
        Return:
            bool
                True (a snapshot can be received without waiting),
                False (otherwise)
        Check if the sampler terminated. If so, get the index of the last snapshot.
        If all snapshots have been received, cancel the last request and set active to False.
        If there is some undelivered snapshot, check if it is ready to be received.
        """
        if not self.sampler_terminated:
            # if the sampler terminated
            if self.request_terminate.Get_status():
                # receive the message
                self.request_terminate.Wait()
                # get the index of the last snapshot that has to be received
                self.max_idx = self.buffer_terminate[0]
                # the sampler terminated but there may be undelivered snapshots
                self.sampler_terminated = True
        # if there are snapshots still to be received (current_idx <= max_idx covers
        # the case where the terminate signal arrives simultaneously with the last
        # snapshot, which would otherwise be silently dropped with strict <)
        if self.current_idx <= self.max_idx:
            if self.request_snapshot.get_status():
                return True
            else:
                return False
        else:
            # all snapshots have been received
            self.all_snapshots_were_received = True
            # cancel the request
            self.request_snapshot.cancel()
            self.request_snapshot.wait()
            self.request_snapshot = None
            return False

    def all_snapshots_received(self):
        return self.all_snapshots_were_received

    def get_snapshot_and_request_new(self) -> List[Any]:
        """
        Receive snapshot and request new.
        """
        snapshot = self.request_snapshot.wait()
        self.current_idx += 1
        self.request_snapshot = self.comm_world.irecv(source=self.rank_sampler, tag=self.current_idx)
        return snapshot

    def terminate(self) -> None:
        """
        Receives all remaining snapshots.
        Receives termination signal if not received yet.
        """
        # wait for termination signal
        # meanwhile, receive_snapshots
        while not self.sampler_terminated:
            if self.snapshot_is_available():
                self.get_snapshot_and_request_new()
        # Termination signal received,
        # the total number of snapshots is known. Drain any remaining
        # in-flight snapshots before cancelling the final posted receive.
        while self.current_idx <= self.max_idx:
            self.get_snapshot_and_request_new()
        self.all_snapshots_were_received = True
        if self.request_snapshot is not None:
            self.request_snapshot.cancel()
            self.request_snapshot.wait()
            self.request_snapshot = None
        self.request_terminate = None
        self.comm_world = None


class Communicator:
    """
    Parent class for communication between samplers and
    full/surrogate solver (and collector, optionally).
    """

    def __init__(self) -> None:
        pass

    def set_parameters(self, parameters: npt.ArrayLike) -> None:
        """
        Sets sample for which the observations will be computed later
        (by the full/surrogate solver).
        """
        pass

    def get_observations(self) -> Tuple[npt.NDArray, int]:
        """
        Gets observations computed by the full/surrogate solver and solver tag.
        """
        raise NotImplementedError

    def send_to_collector(self, data: list) -> None:
        """
        Sends triplets [sample, observations, weight] to the collector.
        Optional.
        """
        pass


class SolverMPI(Communicator):
    # initiated by SAMPLERs
    # communicates with SOLVERS POOL
    # sends parameters (Send, tag=1,2,...)
    # sends signal that sampler terminated (Send, tag=0)
    # receives [observations, solver_tag] pickled (recv, tag=1,2,... = the request tag)
    def __init__(self, conf: Configuration) -> None:
        assert conf.rank_solvers_pool is not None, "rank_solvers_pool is None"
        self.rank_solvers_pool = conf.rank_solvers_pool
        self.max_requests = 1
        self.tag_solver = 0
        self.observations = np.zeros(conf.no_observations)
        self.buffer_empty_signal = np.zeros((1,))
        self.terminated = False
        self.comm_world = MPI.COMM_WORLD
        self.tt_recv = 0
        self.tt_send = 0

    def set_parameters(self, parameters: npt.ArrayLike) -> None:
        self.tag_solver += 1
        tmp = time.time()
        # the solvers pool receives into a float64 buffer; make the sent buffer match (e.g. float32 continued samples)
        self.comm_world.Send(np.ascontiguousarray(parameters, dtype=np.float64), dest=self.rank_solvers_pool, tag=self.tag_solver)
        tmp2 = time.time() - tmp
        self.tt_send = self.tt_send + tmp2

    def get_observations(self, ):
        tmp = time.time()
        # the solvers pool answers with a pickled [observations, solver_tag] payload; the MPI tag
        # of that message is the request tag, never the solver status code (finding 2.4)
        [self.observations, solver_tag] = self.comm_world.recv(source=self.rank_solvers_pool)
        tmp2 = time.time() - tmp
        self.tt_recv = self.tt_recv + tmp2
        return self.observations.copy(), solver_tag

    def terminate(self, ):
        if not self.terminated:
            self.comm_world.Send(self.buffer_empty_signal, dest=self.rank_solvers_pool, tag=TAG_TERMINATE)
            self.terminated = True
        self.comm_world = None
