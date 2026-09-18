#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:00:39 2019

@author: simona
"""

import time
from dataclasses import dataclass
from typing import Any, List, cast

import numpy as np
import numpy.typing as npt

from surrDAMH.distributions.parent import Distribution
from surrDAMH.modules.algorithm_interfaces import (AlgorithmConfiguration,
                                                   EvaluatorProvider,
                                                   ObservationProvider,
                                                   SnapshotCollector)
from surrDAMH.modules.manifest import raw_data_columns, samples_columns
from surrDAMH.modules.monitoring import SamplingOutputMonitor
from surrDAMH.modules.proposals import Proposal
from surrDAMH.stages import Stage


@dataclass
class Sample:
    parameters: npt.NDArray
    observations: npt.NDArray | None = None  # G(parameters)
    observations_approx: npt.NDArray | None = None  # Surrogate(parameters)
    log_likelihood: float | None = None
    log_prior: float | None = None
    log_likelihood_approx: float | None = None
    # Status reported by the solver for this sample:
    #   0 (or any non-negative value) - the solver succeeded, ``observations`` are valid;
    #   < 0                           - the solver FAILED, ``observations`` are invalid
    #                                   (typically zero-filled) and must not be used.
    # A failed sample gets ``log_likelihood = -inf`` (``_compute_log_posterior_terms``) with a
    # finite ``log_prior``, so it is always rejected, and ``_handle_rejection`` does not forward
    # it to the surrogate collector (items B4/C2/C3). A failed INITIAL sample is a fatal error
    # (``_prepare_run`` raises), because the chain could never leave it.
    solver_tag: int = 0

    @property
    def log_posterior(self) -> float | None:
        if self.log_likelihood is None or self.log_prior is None:
            return None
        return self.log_likelihood + self.log_prior

    @property
    def log_posterior_approx(self) -> float | None:
        if self.log_likelihood_approx is None or self.log_prior is None:
            return None
        return self.log_likelihood_approx + self.log_prior

    # copy method:
    def copy(self):
        if self.observations is None:
            observations = None
        else:
            observations = self.observations.copy()
        if self.observations_approx is None:
            observations_approx = None
        else:
            observations_approx = self.observations_approx.copy()

        new_instance = type(self)(parameters=self.parameters.copy(),
                                  observations=observations,
                                  observations_approx=observations_approx,
                                  log_likelihood=self.log_likelihood,
                                  log_prior=self.log_prior,
                                  log_likelihood_approx=self.log_likelihood_approx,
                                  solver_tag=self.solver_tag)
        return new_instance


def sample_carried_to_next_stage(sample: Sample, finished_stage: Stage) -> Sample:
    """
    Sample handed over as the initial sample of the stage following ``finished_stage``.

    After a ``use_only_surrogate`` stage the chain state carries SURROGATE observations and
    log-likelihood. Keeping them would make the next (exact) stage compare an exact ``log L(y)``
    against a surrogate ``log L(x)`` in its first acceptance ratio, and write a surrogate
    log-posterior in its first output row (finding A11). The model-dependent terms are therefore
    dropped here so that ``AlgorithmBase._prepare_run`` re-evaluates the state with the full
    model - one extra solver call per chain and per such stage boundary. ``log_prior`` does not
    depend on the model and is kept (it is recomputed anyway).

    Used identically by ``process_SAMPLER.run_SAMPLER`` and ``runner_local.run_local``.
    """
    if not finished_stage.use_only_surrogate:
        return sample
    carried = sample.copy()
    carried.observations = None
    carried.log_likelihood = None
    carried.observations_approx = None
    carried.log_likelihood_approx = None
    return carried


class AlgorithmBase:
    """
    One sampling stage of one chain: proposal + acceptance test + output writing.

    ``multiplicity`` column of ``samples/<stage>/rank%04d.csv`` (A30, decided 2026-09-17):
    a state's row
    carries ``1 + counter_rejected_current`` -- the state itself plus the proposals rejected
    from it -- EXCEPT for the first row of a stage whose initial state was already written by
    the preceding stage (``initial_sample_is_carried_over=True``, set by both runners for every
    stage after the first whose predecessor had ``save_to_file=True``, including after an
    ``is_excluded`` stage), where the leading ``+1`` is dropped so that the state is counted
    exactly once across the concatenated stages; such a row may have multiplicity 0.

    The same word is used for the snapshot payload sent to the collector
    (``[parameters, observations, multiplicity]``, WS6): ``1 + rejections`` for a state that
    was left, ``0`` for a rejected proposal. What the surrogate updater does with it is
    decided by its ``weighting`` option, see ``surrDAMH.surrogates.parent.Updater``.
    """

    def __init__(self, stage: Stage, proposal: Proposal, initial_sample: Sample, rank_world: int,
                 conf: AlgorithmConfiguration, prior: Distribution, likelihood: Distribution,
                 observation_provider: ObservationProvider, snapshot_collector: SnapshotCollector | None = None,
                 evaluator_provider: EvaluatorProvider | None = None, seed: int = 0,
                 initial_sample_is_carried_over: bool = False) -> None:
        self.stage = stage
        self.proposal = proposal
        self.current = initial_sample
        self.rank_world = rank_world
        self.conf = conf
        self.prior = prior
        self.likelihood = likelihood
        self.observation_provider = observation_provider
        self.snapshot_collector = snapshot_collector
        self.evaluator_provider = evaluator_provider
        self.seed = seed
        if self.snapshot_collector is not None and stage.send_snapshots_to_collector:
            self.send_to_collector = self._send_to_collector
        else:
            self.send_to_collector = self._empty_function
        self.counter_accepted = 0
        self.counter_prerejected = 0
        self.counter_rejected = 0
        self.counter_rejected_current = 0
        # A30: was the initial state of this stage already written (and counted) by the
        # previous stage? If so, its row here must not add the state itself again.
        self.initial_sample_is_carried_over = initial_sample_is_carried_over
        self._first_state_row_pending = True
        self.proposed: Sample
        self._generator = np.random.RandomState(seed)
        self.monitor = SamplingOutputMonitor(
            output_dir=self.conf.output_dir,
            stage=stage,
            basename="rank" + str(self.rank_world).zfill(4) + ".csv",
        )
        # output format v2: the header is written when (and only when) the file is created.
        self.no_parameters = int(self.conf.no_parameters)
        self.no_observations = int(self.conf.no_observations)
        self.monitor.set_header("samples", samples_columns(self.no_parameters))
        self.monitor.set_header("raw_data", raw_data_columns(self.no_parameters, self.no_observations))
        self._prepare_run()
        self._initialize_current_approximation()

    def _evaluate_sample(self, sample: Sample) -> None:
        """Evaluate full-model observations and posterior terms for one sample."""
        self.observation_provider.set_parameters(self.prior.transform(sample.parameters))
        result = self.observation_provider.get_observations()
        if isinstance(result, tuple):
            sample.observations, sample.solver_tag = result
        else:
            sample.observations = result
            sample.solver_tag = 0
        assert sample.observations is not None
        sample.log_likelihood, sample.log_prior = self._compute_log_posterior_terms(
            sample.parameters,
            sample.observations,
            sample.solver_tag,
        )

    def _prepare_run(self) -> None:
        self.time_start = time.time()
        if self.current.observations is None:
            self._evaluate_sample(self.current)
            # checked on the exact log-likelihood only, and only for a freshly evaluated initial sample
            if self.current.log_likelihood is None or not np.isfinite(self.current.log_likelihood):
                raise RuntimeError(
                    f"rank {self.rank_world}: initial sample has non-finite log-likelihood "
                    f"{self.current.log_likelihood} (solver_tag={self.current.solver_tag}); the chain cannot move"
                )
        self.monitor(
            data_name="subchain_stats",
            row=[
                "iteration",
                "subchain_max_length",
                "subchain_accepted",
                "subchain_acceptance_rate",
                "correction_log_ratio",
                "outer_proposed_changed",
                "outer_accepted",
                "rank_world",
            ],
            condition=self.stage.save_to_file and self.stage.algorithm_type == "DAMH",
        )

    def _evaluate_proposed_sample(self) -> None:
        self._evaluate_sample(self.proposed)

    def _propose_new_sample(self, current_parameters: npt.NDArray) -> Sample:
        return Sample(parameters=self.proposal.propose_sample(current_parameters))

    def _emit_current_state(self, multiplicity: float) -> None:
        """Write/forward the current chain state before leaving it.

        ``multiplicity`` is the SURROGATE-TRAINING multiplicity handed to the collector and is
        deliberately left at ``1 + counter_rejected_current`` even at a stage boundary: unlike
        the ``samples`` CSV, the collector never receives a stage's final state (``_finalize_run``
        does not forward it), so there is nothing to double-count there (A30 touches the
        output multiplicity column only, see ``_current_state_row_multiplicity``).
        """
        self._record_current_sample()
        self.send_to_collector(sample=self.current, multiplicity=multiplicity)

    def _observation_block(self, observations: npt.NDArray | None) -> List[Any]:
        """One fixed-width ``conf.no_observations`` block of a ``raw_data`` row, NaN if absent."""
        if observations is None:
            return [np.nan] * self.no_observations
        return list(np.asarray(observations, dtype=float).flatten())

    def _record_proposed_snapshot(self, state_type: str, tag: int,
                                  observations: npt.NDArray | None,
                                  observations_approx: npt.NDArray | None,
                                  log_likelihood: float | None,
                                  log_prior: float | None) -> None:
        """
        Record one proposed state in ``raw_data`` (format v2, rectangular).

        ``observations`` is the EXACT model block and ``observations_approx`` the surrogate
        block; both are always present as ``conf.no_observations`` columns and NaN-filled
        where the corresponding evaluation never happened (a ``prerejected`` proposal has no
        exact observations, an MH stage has no surrogate ones).

        ``log_likelihood``/``log_prior`` are the values the acceptance decision for this row
        actually used, i.e. the EXACT log-likelihood for ``accepted``/``rejected`` rows and
        the SURROGATE one for ``prerejected`` rows (documented in ``docs/outputs.md``).
        """
        if self.conf.save_snapshots_to_file:
            if self.conf.transform_before_saving:
                row: List[Any] = [state_type] + list(self.prior.transform(self.proposed.parameters))
            else:
                row = [state_type] + list(self.proposed.parameters)
            row += [tag]
            row += self._observation_block(observations)
            row += self._observation_block(observations_approx)
            row += [log_likelihood if log_likelihood is not None else np.nan]
            row += [log_prior if log_prior is not None else np.nan]
            self.monitor(data_name="raw_data", row=row)

    def _transition_to_accepted(self) -> None:
        """Update chain state and counters after accepting the proposal."""
        self.counter_accepted += 1
        self.counter_rejected_current = 0
        self.current = self.proposed.copy()

    def _transition_to_rejected(self) -> None:
        """Update chain state and counters after rejecting the proposal."""
        self.counter_rejected += 1
        self.counter_rejected_current += 1

    def _transition_to_prerejected(self) -> None:
        """Update counters when DAMH rejects before full-model evaluation."""
        self.counter_prerejected += 1
        self.counter_rejected_current += 1

    def _handle_acceptance(self) -> None:
        self._emit_current_state(multiplicity=self.counter_rejected_current + 1)
        self._transition_to_accepted()
        self._record_proposed_snapshot(state_type="accepted", tag=self.proposed.solver_tag,
                                       observations=self.proposed.observations,
                                       observations_approx=self.proposed.observations_approx,
                                       log_likelihood=self.proposed.log_likelihood,
                                       log_prior=self.proposed.log_prior)

    def _handle_rejection(self) -> None:
        self._transition_to_rejected()
        if not self.proposed.solver_tag < 0:  # do not train the surrogate on a failed solver run (zero-filled observations)
            self.send_to_collector(sample=self.proposed, multiplicity=0)
        self._record_proposed_snapshot(state_type="rejected", tag=self.proposed.solver_tag,
                                       observations=self.proposed.observations,
                                       observations_approx=self.proposed.observations_approx,
                                       log_likelihood=self.proposed.log_likelihood,
                                       log_prior=self.proposed.log_prior)

    def _compute_log_posterior_terms(self, parameters: npt.NDArray, observation: npt.NDArray,
                                     solver_tag: int = 0) -> tuple[float, float]:
        """Returns ``(log_likelihood, log_prior)`` for one parameter/observation pair."""
        if solver_tag < 0:
            # solver failure: the proposal is rejected through the -inf likelihood ratio,
            # the prior term stays finite so that the prior_part arithmetic remains well defined
            return -np.inf, self.prior.logpdf(parameters)
        log_likelihood = self.likelihood.logpdf(observation)
        log_prior = self.prior.logpdf(parameters)
        return log_likelihood, log_prior

    def _draw_acceptance_decision(self, log_acceptance_probability: float) -> bool:
        temp = self._generator.uniform(0.0, 1.0)
        if np.log(temp) < log_acceptance_probability:
            return True  # accepted
        else:
            return False  # rejected

    def _current_state_row_multiplicity(self) -> int:
        """
        Multiplicity written in the first column of the current state's ``samples`` row,
        consuming the "first row of this stage" flag (A30).

        ``1 + counter_rejected_current`` (the state plus the proposals rejected from it),
        minus the leading 1 for the first row of a stage whose initial state was carried over
        from the previous stage and therefore already counted there. Such a row can have
        multiplicity 0 (the very first proposal of the stage was accepted);
        ``post_processing.decompress`` drops such a row and ``Samples.no_unique_samples``
        does not count it (``np.count_nonzero``). Rule and consequences documented in
        ``docs/outputs.md``.
        """
        multiplicity = 1 + self.counter_rejected_current
        if self._first_state_row_pending and self.initial_sample_is_carried_over:
            multiplicity -= 1
        self._first_state_row_pending = False
        return multiplicity

    def _record_current_sample(self) -> None:
        multiplicity = self._current_state_row_multiplicity()
        if self.conf.transform_before_saving:
            row: List[Any] = [multiplicity] + list(self.prior.transform(self.current.parameters))
        else:
            row: List[Any] = [multiplicity] + list(self.current.parameters)
        row.append(self.current.log_posterior)
        self.monitor(data_name="samples", row=row, condition=self.stage.save_to_file)

    def _send_to_collector(self, sample: Sample, multiplicity: float) -> None:
        """Forwards one snapshot ``[parameters, observations, multiplicity]`` to the collector."""
        parameters = sample.parameters.copy()
        assert sample.observations is not None
        observations = sample.observations.copy()
        if self.conf.transform_before_surrogate:
            parameters = self.prior.transform(parameters)
        assert self.snapshot_collector is not None
        self.snapshot_collector.send_to_collector([parameters, observations, multiplicity])

    def _empty_function(self, **kw) -> None:
        return

    def _finalize_run(self) -> None:
        self._record_current_sample()
        self.monitor(data_name="notes", row=["accepted", "rejected", "pre-rejected", "sum", "seed"], condition=self.stage.save_to_file)
        no_all = self.counter_accepted + self.counter_rejected + self.counter_prerejected
        notes = [self.counter_accepted, self.counter_rejected, self.counter_prerejected, no_all, self.seed]
        self.monitor(data_name="notes", row=notes, condition=self.stage.save_to_file)
        self.monitor.close_files()

    def _get_surrogate_observations(self, parameters0: npt.NDArray, parameters1: npt.NDArray | None = None):
        if self.conf.transform_before_surrogate:
            par0_tr = self.prior.transform(parameters0.copy())
            argument0 = [par0_tr]
            if parameters1 is not None:
                par1_tr = self.prior.transform(parameters1.copy())
                argument1 = [par1_tr]
        else:
            argument0 = [parameters0.copy()]
            if parameters1 is not None:
                argument1 = [parameters1.copy()]
        assert self.surrogate_evaluator is not None
        # the evaluator contract is (n, no_observations) for every n (WS6), so a one-point
        # batch is indexed with [0] to get the (no_observations,) vector the likelihood and
        # Sample.observations_approx expect -- no defensive flattening needed any more
        if parameters1 is None:
            return self.surrogate_evaluator(np.array(argument0))[0]
        else:
            res0 = self.surrogate_evaluator(np.array(argument0))
            res1 = self.surrogate_evaluator(np.array(argument1))
            return res0[0], res1[0]

    def _initialize_current_approximation(self) -> None:
        if self.evaluator_provider is None:
            return
        self.surrogate_evaluator = self.evaluator_provider.evaluator
        if self.surrogate_evaluator is None:
            self.surrogate_evaluator = self.evaluator_provider.get_evaluator()
            self.evaluator_provider.request_evaluator()
        self.current.observations_approx = cast(npt.NDArray, self._get_surrogate_observations(self.current.parameters))
        # assert self.current.observations_approx is not None
        self.current.log_likelihood_approx, self.current.log_prior = self._compute_log_posterior_terms(
            self.current.parameters,
            self.current.observations_approx,
        )
        if self.conf.use_surrogate_gradients:
            log_likelihood_gradient_function = lambda x: self._compute_surrogate_log_likelihood_gradient(x)
            log_prior_gradient_function = lambda x: self._compute_log_prior_gradient(x)
            self.proposal.set_gradient_functions(log_likelihood_gradient_function, log_prior_gradient_function)

    def _compute_surrogate_log_likelihood_gradient(self, parameters: npt.NDArray) -> npt.NDArray:
        # calculates likelihood part of grad(U(q)). i.e. grad(-log_likelihood)
        if self.conf.transform_before_surrogate:
            # raise not implemented error
            raise NotImplementedError("Gradient for transform_before_surrogate=True is not implemented")
            argument = self.prior.transform(parameters.copy())
        else:
            argument = parameters.copy()
        assert self.surrogate_evaluator is not None
        evaluation = self.surrogate_evaluator(np.array([argument]))[0]
        vector = -self.likelihood.grad_logpdf(evaluation)
        try:
            gradient, _ = self.surrogate_evaluator.vjp(np.array(argument), vector)
            return gradient
        except NotImplementedError:
            jacobian, evaluation = self.surrogate_evaluator.jacobian(np.array(argument))
            # TODO: do not compute again
            return - jacobian.T @ self.likelihood.grad_logpdf(evaluation)

    def _compute_log_prior_gradient(self, parameters: npt.NDArray) -> npt.NDArray:
        # calculates prior part of grad(U(q)). i.e. grad(-log_prior)
        return -self.prior.grad_logpdf(parameters) 


class Algorithm_MH(AlgorithmBase):  # initiated by SAMPLERs
    def run(self) -> None:
        max_steps = min(self.stage.max_samples, self.stage.max_evaluations)
        for i in range(max_steps):
            self.proposal.choose_group()  # only for block proposal, does nothing for non-block proposal
            self.proposed = self._propose_new_sample(self.current.parameters)
            self._evaluate_proposed_sample()
            assert self.proposed.log_likelihood is not None
            assert self.current.log_likelihood is not None
            assert self.proposed.log_prior is not None
            assert self.current.log_prior is not None
            likelihood_part, prior_part = self.proposal.get_log_acceptance_probability(
                self.proposed.log_likelihood,
                self.current.log_likelihood,
                self.proposed.log_prior,
                self.current.log_prior,
            )
            log_acceptance_prob_exact = likelihood_part + prior_part
            self.proposal.adapt(proposed_sample=self.proposed.parameters, log_acceptance_probability=log_acceptance_prob_exact)
            if self._draw_acceptance_decision(log_acceptance_prob_exact):
                self._handle_acceptance()
            else:
                self._handle_rejection()
            if time.time() - self.time_start > self.stage.time_limit:
                print("SAMPLER at rank", self.rank_world, "time limit ", self.stage.time_limit, " reached - loop", i, flush=True)
                break
            print(f"Progress: {i}, accepted: {self.counter_accepted}, rejected: {self.counter_rejected}", end="\r", flush=True)
        self._finalize_run()


class Algorithm_DAMH(AlgorithmBase):  # initiated by SAMPLERs
    def _refresh_surrogate_evaluator_if_needed(self) -> bool:
        assert self.evaluator_provider is not None
        if self.stage.surrogate_model_updates and self.evaluator_provider.evaluator_is_available():
            self.surrogate_evaluator = self.evaluator_provider.get_evaluator()
            self.evaluator_provider.request_evaluator()
            return True
        return False

    def _evaluate_surrogate_transition(self, subchain_current: Sample, subchain_proposed: Sample, surrogate_evaluator_changed: bool) -> None:
        """
        Fill in the surrogate terms of one sub-chain transition ``subchain_current -> subchain_proposed``.

        Sets ``observations_approx``, ``log_likelihood_approx`` and ``log_prior`` of
        ``subchain_proposed`` (and of ``subchain_current``), so that the caller can form the
        sub-chain MH ratio w.r.t. the surrogate posterior.

        ``surrogate_evaluator_changed=True`` means "the evaluator was just replaced": the
        surrogate terms cached on ``subchain_current`` and on ``self.current`` refer to the old
        surrogate and are re-scored here. It is passed only on the first iteration of a sub-chain
        (the evaluator is frozen for the rest of it), and only if the evaluator really changed.
        With ``False``, only the newly proposed state has to be evaluated.

        If ``conf.state_dependent_approximation`` is True the surrogate is used as an additive
        correction around ``self.current`` instead of directly; that option is UNVERIFIED for
        ``subchain_max_length > 1`` (finding 1.1) and is left unchanged.
        """
        if surrogate_evaluator_changed:
            subchain_current.observations_approx, subchain_proposed.observations_approx = cast(
                tuple[npt.NDArray, npt.NDArray],
                self._get_surrogate_observations(subchain_current.parameters, subchain_proposed.parameters),
            )
            self.current.observations_approx = cast(npt.NDArray, self._get_surrogate_observations(self.current.parameters))
            # assert self.current.observations_approx is not None
            self.current.log_likelihood_approx, _ = self._compute_log_posterior_terms(
                self.current.parameters,
                self.current.observations_approx,
            )
        else:
            subchain_proposed.observations_approx = cast(
                npt.NDArray,
                self._get_surrogate_observations(subchain_proposed.parameters),
            )
        # assert subchain_proposed.observations_approx is not None

        if self.conf.state_dependent_approximation:
            assert self.current.observations is not None
            # assert self.current.observations_approx is not None
            observations_approx_shifted = (
                subchain_proposed.observations_approx
                + self.current.observations
                - self.current.observations_approx
            )
            subchain_proposed.log_likelihood_approx, subchain_proposed.log_prior = self._compute_log_posterior_terms(
                subchain_proposed.parameters,
                observations_approx_shifted,
            )
            subchain_current.log_likelihood_approx, subchain_current.log_prior = self._compute_log_posterior_terms(
                subchain_current.parameters,
                self.current.observations,
            )
        else:
            assert subchain_current.observations_approx is not None
            subchain_proposed.log_likelihood_approx, subchain_proposed.log_prior = self._compute_log_posterior_terms(
                subchain_proposed.parameters,
                subchain_proposed.observations_approx,
            )
            subchain_current.log_likelihood_approx, subchain_current.log_prior = self._compute_log_posterior_terms(
                subchain_current.parameters,
                subchain_current.observations_approx,
            )

    def _propose_new_sample_using_subchain(self) -> tuple[Sample, int, float]:
        """
        Run one sub-chain of at most ``stage.subchain_max_length`` MH steps that use only the
        surrogate, starting from ``self.current``.

        Returns ``(subchain_current, counter_subchain_accepted, correction_log_ratio)``:
        the sub-chain end state (the proposal for the outer/exact MH step), the number of
        accepted sub-chain steps, and the sum of the surrogate log-likelihood ratios of those
        accepted steps.

        The surrogate evaluator is FROZEN for the whole sub-chain: it is refreshed once, here,
        before the loop (finding 1.2). That is what makes ``correction_log_ratio`` telescope to
        ``log L~(y) - log L~(x)`` for a single surrogate, as the outer correction in ``run()``
        requires; refreshing inside the loop would sum ratios taken under different surrogates.
        The evaluator that arrives while a sub-chain is running is picked up at the start of the
        next one (one poll per outer step; the request/poll protocol with the collector is
        unchanged - at most one outstanding request at a time).
        """
        counter_subchain_accepted = 0
        subchain_current = self.current.copy()
        correction_log_ratio = 0.0
        # refresh once per sub-chain, then keep the surrogate fixed until the sub-chain ends:
        bool_evaluator_changed = self._refresh_surrogate_evaluator_if_needed()
        for _ in range(self.stage.subchain_max_length):
            subchain_proposed = self._propose_new_sample(subchain_current.parameters)
            self._evaluate_surrogate_transition( # evaluate surrogate for current, subchain_surrent, subchain_proposed
                subchain_current=subchain_current,
                subchain_proposed=subchain_proposed,
                surrogate_evaluator_changed=bool_evaluator_changed,
            )
            # self.current and the sub-chain start were re-scored with the new evaluator above;
            # from here on the surrogate does not change, so only the proposal has to be evaluated:
            bool_evaluator_changed = False
            assert subchain_proposed.log_likelihood_approx is not None
            assert subchain_current.log_likelihood_approx is not None
            assert subchain_proposed.log_prior is not None
            assert subchain_current.log_prior is not None
            likelihood_part, prior_part = self.proposal.get_log_acceptance_probability(
                subchain_proposed.log_likelihood_approx,
                subchain_current.log_likelihood_approx,
                subchain_proposed.log_prior,
                subchain_current.log_prior,
            )
            log_acceptance_prob_approx = likelihood_part + prior_part
            if self._draw_acceptance_decision(log_acceptance_prob_approx):
                correction_log_ratio += likelihood_part
                counter_subchain_accepted += 1
                subchain_current = subchain_proposed.copy()

        return subchain_current, counter_subchain_accepted, correction_log_ratio

    def run(self) -> None:
        for i in range(self.stage.max_samples):
            self.proposal.choose_group()  # only for block proposal, does nothing for non-block proposal
            subchain_current, counter_subchain, correction_log_ratio = self._propose_new_sample_using_subchain()
            self.proposed = subchain_current.copy()
            outer_accepted = False
            if counter_subchain > 0:  # at least one proposal of the subchain was accepted
                self._evaluate_proposed_sample()
                assert self.proposed.log_likelihood is not None
                assert self.current.log_likelihood is not None
                assert self.proposed.log_prior is not None
                assert self.current.log_prior is not None
                likelihood_part, prior_part = self.proposal.get_log_acceptance_probability(
                    self.proposed.log_likelihood,
                    self.current.log_likelihood,
                    self.proposed.log_prior,
                    self.current.log_prior,
                )
                log_acceptance_prob_exact = likelihood_part + prior_part
                self.proposal.adapt(proposed_sample=self.proposed.parameters, log_acceptance_probability=log_acceptance_prob_exact)  # TODO
                assert self.proposed.log_likelihood_approx is not None
                assert self.current.log_likelihood_approx is not None
                # Delayed-acceptance correction. The sub-chain is an MH kernel that is reversible
                # w.r.t. the surrogate posterior pi~, so its transition density satisfies
                #     Q(y -> x) / Q(x -> y) = pi~(y) / pi~(x) = [L~(y) prior(y)] / [L~(x) prior(x)].
                # The outer MH ratio is pi(y) Q(y -> x) / (pi(x) Q(x -> y)) = L(y) L~(x) / (L(x) L~(y)),
                # i.e. the prior cancels and
                #     log alpha = [log L(y) - log L(x)] - [log L~(y) - log L~(x)].
                # correction_log_ratio accumulates the surrogate likelihood log-ratios of the accepted
                # sub-chain steps, which telescopes to log L~(y) - log L~(x) for a surrogate that is
                # FIXED during the sub-chain. This is guaranteed: _propose_new_sample_using_subchain
                # refreshes the evaluator once, before the sub-chain starts, and freezes it until the
                # sub-chain ends (finding 1.2, WS3). Remaining caveat: with
                # conf.state_dependent_approximation=True the surrogate terms are state-dependent
                # shifts around self.current, for which this derivation has not been verified for
                # subchain_max_length > 1 (finding 1.1, library_notes/06; option unverified by design).
                exact_likelihood_log_ratio = self.proposed.log_likelihood - self.current.log_likelihood
                accepted = self._draw_acceptance_decision(exact_likelihood_log_ratio - correction_log_ratio)
                if accepted:
                    outer_accepted = True
                    self._handle_acceptance()
                else:
                    self._handle_rejection()
            else:  # proposed sample is the same as current sample, sample is automatically accepted, the chain remains here
                self._transition_to_prerejected()
                # v2: the exact model was never called for a prerejected proposal, so its
                # exact observation block is NaN and the logged log-likelihood is the
                # SURROGATE one the sub-chain acceptance test used (docs/outputs.md).
                self._record_proposed_snapshot(state_type="prerejected", tag=0,
                                               observations=None,
                                               observations_approx=self.proposed.observations_approx,
                                               log_likelihood=self.proposed.log_likelihood_approx,
                                               log_prior=self.proposed.log_prior)
            self.monitor(
                data_name="subchain_stats",
                row=[
                    i,
                    self.stage.subchain_max_length,
                    counter_subchain,
                    counter_subchain / self.stage.subchain_max_length,
                    correction_log_ratio,
                    int(counter_subchain > 0),
                    int(outer_accepted),
                    self.rank_world,
                ],
                condition=self.stage.save_to_file and self.stage.algorithm_type == "DAMH",
            )
            if time.time() - self.time_start > self.stage.time_limit:
                break
            print(f"Progress: {i}, accepted: {self.counter_accepted}, rejected: {self.counter_rejected}, prerejected: {self.counter_prerejected}", end="\r", flush=True)
            if (self.counter_rejected + self.counter_accepted) >= self.stage.max_evaluations:
                break
        self._finalize_run()


