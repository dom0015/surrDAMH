#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:00:39 2019

@author: simona
"""

import time
import warnings
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

#: ``Sample.solver_tag`` of a proposal that was never handed to the solver because its
#: parameters are not all finite (2026-09-20, see ``AlgorithmBase._reject_nonfinite_proposal``).
SOLVER_TAG_NONFINITE_PROPOSAL = -2


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
    #   -2 (SOLVER_TAG_NONFINITE_PROPOSAL) - the proposal itself was not finite, so the solver
    #                                   was NEVER CALLED for it; ``observations`` are zero-filled.
    #                                   Reserved by the library: a user solver must not return it.
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
        # how many times _refresh_surrogate_evaluator_if_needed really replaced the evaluator
        self.counter_evaluator_refreshes = 0
        # non-finite proposals rejected without any model evaluation (2026-09-20, item A);
        # counted for the outer chain and for every sub-chain step of a DAMH stage
        self.counter_nonfinite_proposals = 0
        self._nonfinite_proposal_warning_emitted = False
        # evaluators that were refused because they predicted non-finite values (item D)
        self.counter_nonfinite_evaluators = 0
        self._nonfinite_evaluator_warning_emitted = False
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

    @staticmethod
    def _parameters_are_finite(sample: Sample) -> bool:
        return bool(np.all(np.isfinite(sample.parameters)))

    def _warn_about_nonfinite_proposal(self) -> None:
        """One ``RuntimeWarning`` per stage, naming the proposal that produced the outlier."""
        if self._nonfinite_proposal_warning_emitted:
            return
        self._nonfinite_proposal_warning_emitted = True
        details = ""
        step_size = getattr(self.proposal, "step_size", None)
        num_steps = getattr(self.proposal, "num_steps", None)
        if step_size is not None and num_steps is not None:
            details = f" (step_size={step_size!r}, num_steps={num_steps!r})"
        warnings.warn(
            f"rank {self.rank_world}, stage {self.stage.name}: proposal "
            f"{type(self.proposal).__name__}{details} produced a NON-FINITE sample; it is rejected "
            "without evaluating the full model and is not sent to the surrogate collector "
            "(further occurrences in this stage are counted, not warned about)",
            RuntimeWarning,
            stacklevel=2,
        )

    def _reject_nonfinite_proposal(self, sample: Sample) -> None:
        """
        Mark a non-finite proposal as "not evaluated" instead of calling the forward model.

        A divergent leapfrog trajectory (or any other proposal that ran off to ``inf``/``nan``)
        is rejected with probability 1 anyway, but evaluating it used to cost one full-model
        solve -- which can itself overflow -- and, worse, the resulting snapshot was forwarded
        to the collector and destroyed the neural-network surrogate (all subsequent predictions
        NaN, 2026-09-20). ``solver_tag = SOLVER_TAG_NONFINITE_PROPOSAL`` is negative, so the
        existing failed-solver handling gives ``log_likelihood = -inf`` and
        ``_handle_rejection`` keeps the sample away from the collector.
        """
        sample.observations = np.zeros(self.no_observations)
        sample.solver_tag = SOLVER_TAG_NONFINITE_PROPOSAL
        sample.log_likelihood, sample.log_prior = self._compute_log_posterior_terms(
            sample.parameters,
            sample.observations,
            sample.solver_tag,
        )
        self.counter_nonfinite_proposals += 1
        self._warn_about_nonfinite_proposal()

    def _evaluate_proposed_sample(self) -> None:
        if not self._parameters_are_finite(self.proposed):
            self._reject_nonfinite_proposal(self.proposed)
            return
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

    def _write_adaptation_stats(self) -> None:
        """
        Dump the proposal's per-period adaptation trace to ``adaptive_stats/<stage>/rank%04d.csv``
        (2026-09-20, `16` §5 item 2 "persist ... a per-period adaptive_stats.csv").

        An adaptive proposal appends one tuple per completed adaptation period to
        ``adaptation_stats_rows`` and names the columns in ``adaptation_stats_header``; a
        non-adaptive proposal has neither, so no file is created for its stage (the reader then
        reports ``adaptive_stats[stage] is None``). The header is written as the first row, the
        same way ``subchain_stats``'s header is, and ``rank_world`` is appended to every row so
        that the per-rank files stay distinguishable once a reader concatenates them.
        """
        rows = getattr(self.proposal, "adaptation_stats_rows", None)
        if not rows:
            return
        header = list(getattr(self.proposal, "adaptation_stats_header", [])) + ["rank_world"]
        self.monitor(data_name="adaptive_stats", row=header, condition=self.stage.save_to_file)
        for row in rows:
            self.monitor(data_name="adaptive_stats", row=list(row) + [self.rank_world],
                         condition=self.stage.save_to_file)

    def _finalize_run(self) -> None:
        self._record_current_sample()
        self._write_adaptation_stats()
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
            self._install_gradient_functions()

    def _install_gradient_functions(self) -> None:
        """
        Hand the proposal the surrogate-based gradient callables (grad of -log L / -log prior).

        Both closures read ``self.surrogate_evaluator`` when they are called, so replacing the
        evaluator is enough to change the gradients; they are re-installed after a refresh
        anyway (``Algorithm_MH.run``), so that a proposal which pre-processes or caches what it
        is given -- ``BlockProposal`` re-wraps them per group -- sees the change too.
        """
        log_likelihood_gradient_function = lambda x: self._compute_surrogate_log_likelihood_gradient(x)
        log_prior_gradient_function = lambda x: self._compute_log_prior_gradient(x)
        self.proposal.set_gradient_functions(log_likelihood_gradient_function, log_prior_gradient_function)

    def _refresh_surrogate_evaluator_if_needed(self) -> bool:
        """
        Poll the provider once and install a newer surrogate evaluator if one has arrived.

        Returns ``True`` iff the evaluator was replaced. Shared by ``Algorithm_DAMH`` (once per
        sub-chain, so that the sub-chain's correction ratio telescopes under a FIXED surrogate)
        and, since WS7, by ``Algorithm_MH`` (once per iteration, for the proposal's gradients
        only). The request/poll protocol is unchanged: at most one outstanding request, and a
        new one is posted as soon as the previous evaluator is taken.

        An arriving evaluator that predicts non-finite values at the current chain state is
        REFUSED (2026-09-20, item D): the previous one stays installed, the next request is
        still posted, and ``False`` is returned. The very first evaluator is installed even
        then, because nothing better exists.
        """
        if self.evaluator_provider is None:
            return False
        if self.stage.surrogate_model_updates and self.evaluator_provider.evaluator_is_available():
            previous_evaluator = getattr(self, "surrogate_evaluator", None)
            candidate = self.evaluator_provider.get_evaluator()
            self.evaluator_provider.request_evaluator()
            if not self._evaluator_is_finite(candidate):
                self.counter_nonfinite_evaluators += 1
                self._warn_about_nonfinite_evaluator(installed_anyway=previous_evaluator is None)
                if previous_evaluator is not None:
                    # keep sampling with the last good surrogate; the request above stays posted,
                    # so a later, healthy evaluator is still picked up (2026-09-20, item D)
                    return False
                # nothing better exists yet: install it anyway, as before
            self.surrogate_evaluator = candidate
            self.counter_evaluator_refreshes += 1
            return True
        return False

    def _evaluator_is_finite(self, evaluator) -> bool:
        """
        Probe a freshly arrived evaluator at the current chain state (2026-09-20, item D).

        One surrogate call, plus one ``vjp`` with a unit vector when the proposal needs
        gradients, is enough to detect the failure mode this exists for: a neural-network
        updater whose weights went NaN publishes an evaluator that returns NaN everywhere,
        and installing it turns every subsequent proposal into NaN. Evaluators that cannot
        differentiate (``NotImplementedError``) are judged on their value alone.
        """
        parameters = np.asarray(self.current.parameters, dtype=float)
        if self.conf.transform_before_surrogate:
            parameters = np.asarray(self.prior.transform(parameters.copy()), dtype=float)
        prediction = np.asarray(evaluator(np.array([parameters])), dtype=float)
        if not np.all(np.isfinite(prediction)):
            return False
        needs_gradients = bool(getattr(self.proposal, "needs_gradients", False)
                               and self.conf.use_surrogate_gradients
                               and evaluator.supports_gradients())
        if needs_gradients:
            try:
                gradient, evaluation = evaluator.vjp(parameters, np.ones(self.no_observations))
            except NotImplementedError:
                return True
            if not (np.all(np.isfinite(gradient)) and np.all(np.isfinite(evaluation))):
                return False
        return True

    def _warn_about_nonfinite_evaluator(self, installed_anyway: bool) -> None:
        """One ``RuntimeWarning`` per stage about a surrogate that predicts non-finite values."""
        if self._nonfinite_evaluator_warning_emitted:
            return
        self._nonfinite_evaluator_warning_emitted = True
        tail = ("no previous evaluator exists, so it is installed anyway" if installed_anyway
                else "keeping the previous evaluator")
        warnings.warn(
            f"rank {self.rank_world}, stage {self.stage.name}: the surrogate evaluator received "
            f"from the collector is NON-FINITE at the current chain state; {tail} (further "
            "occurrences in this stage are counted, not warned about)",
            RuntimeWarning,
            stacklevel=2,
        )

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
    def _gradient_surrogate_refresh_is_enabled(self) -> bool:
        """
        Should this MH stage re-poll the collector for a newer gradient surrogate (WS7)?

        Only when the stage opted in (``Stage.surrogate_model_updates``, which ``__post_init__``
        only lets an MH stage keep if its proposal needs gradients), the proposal really needs
        gradients, gradients are wired at all and an evaluator provider exists. Without the
        opt-in this is False and the stage keeps the single evaluator fetched by
        ``_initialize_current_approximation`` for its whole run, exactly as before WS7.
        """
        return bool(self.stage.surrogate_model_updates
                    and getattr(self.proposal, "needs_gradients", False)
                    and self.conf.use_surrogate_gradients
                    and self.evaluator_provider is not None)

    def run(self) -> None:
        max_steps = min(self.stage.max_samples, self.stage.max_evaluations)
        refresh_gradient_surrogate = self._gradient_surrogate_refresh_is_enabled()
        for i in range(max_steps):
            if refresh_gradient_surrogate and self._refresh_surrogate_evaluator_if_needed():
                # only the proposal's gradient field changes; the acceptance test below uses
                # the exact model only, so the chain remains exact for any surrogate (WS7)
                self._install_gradient_functions()
            self.proposal.choose_group()  # only for block proposal, does nothing for non-block proposal
            self.proposed = self._propose_new_sample(self.current.parameters)
            self._evaluate_proposed_sample()
            assert self.proposed.log_likelihood is not None
            assert self.current.log_likelihood is not None
            assert self.proposed.log_prior is not None
            assert self.current.log_prior is not None
            if self.proposed.solver_tag == SOLVER_TAG_NONFINITE_PROPOSAL:
                # The proposal is not finite (2026-09-20, item A). get_log_acceptance_probability
                # is skipped on purpose: a Hamiltonian's momentum term would be NaN and the
                # decision below would then depend on NaN comparisons. Probability 0 is also the
                # correct signal for the dual averaging -- Stan scores a divergent transition as
                # alpha = 0 -- so adapt() below receives -inf.
                log_acceptance_prob_exact = -np.inf
            else:
                likelihood_part, prior_part = self.proposal.get_log_acceptance_probability(
                    self.proposed.log_likelihood,
                    self.current.log_likelihood,
                    self.proposed.log_prior,
                    self.current.log_prior,
                )
                log_acceptance_prob_exact = likelihood_part + prior_part
            proposed_parameters = self.proposed.parameters
            if self._draw_acceptance_decision(log_acceptance_prob_exact):
                self._handle_acceptance()
            else:
                self._handle_rejection()
            # adapt() is called AFTER the decision (2026-09-20) so that ``current_sample`` is the
            # POST-decision chain state, which is what a Haario-style covariance estimator needs.
            # The move changes no random stream: adapt() draws no random numbers and its other
            # inputs do not depend on the decision (evidence in CHANGELOG, "Behaviour changes").
            self.proposal.adapt(proposed_sample=proposed_parameters,
                                log_acceptance_probability=log_acceptance_prob_exact,
                                current_sample=self.current.parameters)
            if time.time() - self.time_start > self.stage.time_limit:
                print("SAMPLER at rank", self.rank_world, "time limit ", self.stage.time_limit, " reached - loop", i, flush=True)
                break
            print(f"Progress: {i}, accepted: {self.counter_accepted}, rejected: {self.counter_rejected}", end="\r", flush=True)
        if getattr(self.proposal, "needs_gradients", False):
            # observable proof of what this stage did with its gradient surrogate: 0 for a stage
            # that did not opt in (one evaluator for the whole stage), >0 when it refreshed (WS7)
            print("Stage", self.stage.name, "at MPI rank", self.rank_world,
                  "- gradient surrogate refreshes:", self.counter_evaluator_refreshes, flush=True)
        self._finalize_run()


class Algorithm_DAMH(AlgorithmBase):  # initiated by SAMPLERs
    # _refresh_surrogate_evaluator_if_needed lives in AlgorithmBase since WS7 (shared with
    # Algorithm_MH); its behaviour here is unchanged.

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

        The surrogate posterior is used directly, never as a state-dependent additive
        correction around ``self.current``: the delayed-acceptance argument in ``run()``
        requires a fixed, state-independent surrogate density (the removed
        ``conf.state_dependent_approximation`` option violated that, finding 1.1).
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

        assert subchain_current.observations_approx is not None
        subchain_proposed.log_likelihood_approx, subchain_proposed.log_prior = self._compute_log_posterior_terms(
            subchain_proposed.parameters,
            subchain_proposed.observations_approx,
        )
        subchain_current.log_likelihood_approx, subchain_current.log_prior = self._compute_log_posterior_terms(
            subchain_current.parameters,
            subchain_current.observations_approx,
        )

    def _propose_new_sample_using_subchain(self) -> tuple[Sample, int, float, list[float]]:
        """
        Run one sub-chain of at most ``stage.subchain_max_length`` MH steps that use only the
        surrogate, starting from ``self.current``.

        Returns ``(subchain_current, counter_subchain_accepted, correction_log_ratio,
        subchain_log_acceptance_probabilities)``: the sub-chain end state (the proposal for the
        outer/exact MH step), the number of accepted sub-chain steps, the sum of the surrogate
        log-likelihood ratios of those accepted steps, and the per-step log acceptance
        probabilities the sub-chain computed against the surrogate posterior (one per step, in
        order; they are computed for the sub-chain decision anyway and are what
        ``Proposal.adapt`` dual-averages a Hamiltonian step size on, `16` §5 item 4).

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
        subchain_log_acceptance_probabilities: list[float] = []
        # refresh once per sub-chain, then keep the surrogate fixed until the sub-chain ends:
        bool_evaluator_changed = self._refresh_surrogate_evaluator_if_needed()
        for _ in range(self.stage.subchain_max_length):
            subchain_proposed = self._propose_new_sample(subchain_current.parameters)
            if not self._parameters_are_finite(subchain_proposed):
                # Non-finite sub-chain proposal (2026-09-20, item A): the surrogate is not
                # called for it (a divergent leapfrog would otherwise keep paying for NaN
                # gradient evaluations) and the step is scored as acceptance probability 0.
                # The sub-chain simply stays where it is, so its end state -- the outer
                # proposal -- is always finite.
                self.counter_nonfinite_proposals += 1
                self._warn_about_nonfinite_proposal()
                subchain_log_acceptance_probabilities.append(-np.inf)
                # the draw is kept so that a sub-chain step consumes exactly one uniform
                # whether or not the proposal was finite; log(u) < -inf is never true
                self._draw_acceptance_decision(-np.inf)
                continue
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
            subchain_log_acceptance_probabilities.append(log_acceptance_prob_approx)
            if self._draw_acceptance_decision(log_acceptance_prob_approx):
                correction_log_ratio += likelihood_part
                counter_subchain_accepted += 1
                subchain_current = subchain_proposed.copy()

        return (subchain_current, counter_subchain_accepted, correction_log_ratio,
                subchain_log_acceptance_probabilities)

    def run(self) -> None:
        for i in range(self.stage.max_samples):
            self.proposal.choose_group()  # only for block proposal, does nothing for non-block proposal
            (subchain_current, counter_subchain, correction_log_ratio,
             subchain_log_acceptance_probabilities) = self._propose_new_sample_using_subchain()
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
                proposed_parameters = self.proposed.parameters
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
                # sub-chain ends (finding 1.2, WS3). The derivation also needs the surrogate
                # density to be state-INdependent, which is why the state-dependent shift option
                # was removed (finding 1.1): a surrogate re-centred on the current outer state
                # makes pi~ change with x, and Q(y -> x) / Q(x -> y) no longer equals pi~(y)/pi~(x).
                exact_likelihood_log_ratio = self.proposed.log_likelihood - self.current.log_likelihood
                accepted = self._draw_acceptance_decision(exact_likelihood_log_ratio - correction_log_ratio)
                if accepted:
                    outer_accepted = True
                    self._handle_acceptance()
                else:
                    self._handle_rejection()
                # after the outer decision, so that current_sample is the post-decision state
                self.proposal.adapt(
                    proposed_sample=proposed_parameters,
                    log_acceptance_probability=log_acceptance_prob_exact,
                    current_sample=self.current.parameters,
                    subchain_log_acceptance_probabilities=subchain_log_acceptance_probabilities)
            else:  # proposed sample is the same as current sample, sample is automatically accepted, the chain remains here
                # The adaptive proposal must see this iteration too, scored as acceptance
                # probability 0: adapting only on iterations whose sub-chain moved feeds it the
                # conditional second-stage rate, which tends to 1 as the surrogate improves and
                # has no optimum -- the adaptation then inflates the scale without bound
                # (library_notes/16_adaptivity_options_research_2026-09-20.md §2 item 1, §4.2).
                # `self.proposed` equals the current state here, and so does the post-decision
                # state passed as `current_sample`; only the acceptance-rate feedback changes.
                # The sub-chain's own per-step acceptance probabilities ARE passed on (they
                # exist even though the sub-chain never moved) -- that is what the Hamiltonian
                # dual averaging adapts on. No-op for non-adaptive proposals.
                self.proposal.adapt(
                    proposed_sample=self.proposed.parameters,
                    log_acceptance_probability=-np.inf,
                    current_sample=self.current.parameters,
                    subchain_log_acceptance_probabilities=subchain_log_acceptance_probabilities)
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


