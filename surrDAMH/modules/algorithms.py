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
from surrDAMH.modules.monitoring import SamplingOutputMonitor
from surrDAMH.modules.proposals import Proposal
from surrDAMH.stages import Stage

from surrDAMH.modules.tools import evaluate_on_a_grid

@dataclass
class Sample:
    parameters: npt.NDArray
    observations: npt.NDArray | None = None  # G(parameters)
    observations_approx: npt.NDArray | None = None  # Surrogate(parameters)
    log_likelihood: float | None = None
    log_prior: float | None = None
    log_likelihood_approx: float | None = None
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


class AlgorithmBase:
    def __init__(self, stage: Stage, proposal: Proposal, initial_sample: Sample, rank_world: int,
                 conf: AlgorithmConfiguration, prior: Distribution, likelihood: Distribution,
                 observation_provider: ObservationProvider, snapshot_collector: SnapshotCollector | None = None,
                 evaluator_provider: EvaluatorProvider | None = None, seed: int = 0) -> None:
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
        self.proposed: Sample
        self._generator = np.random.RandomState(seed)
        self.monitor = SamplingOutputMonitor(
            output_dir=self.conf.output_dir,
            stage=stage,
            basename="rank" + str(self.rank_world).zfill(4) + ".csv",
        )
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

    def _emit_current_state(self, weight: float) -> None:
        """Write/forward the current chain state before leaving it."""
        self._record_current_sample()
        self.send_to_collector(sample=self.current, weight=weight)

    def _record_proposed_snapshot(self, state_type: str, tag: int, observations: npt.NDArray | None) -> None:
        """Record information about the currently proposed state (with observations or approximate observations)."""
        if self.conf.save_snapshots_to_file:
            if self.conf.transform_before_saving:
                row = [state_type] + list(self.prior.transform(self.proposed.parameters))
            else:
                row = [state_type] + list(self.proposed.parameters)
            row += [tag]
            if observations is not None:
                row += list(observations.flatten())
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
        self._emit_current_state(weight=self.counter_rejected_current + 1)
        self._transition_to_accepted()
        self._record_proposed_snapshot(state_type="accepted", tag=self.proposed.solver_tag, observations=self.proposed.observations)

    def _handle_rejection(self) -> None:
        self._transition_to_rejected()
        if not self.current.solver_tag < 0:
            self.send_to_collector(sample=self.proposed, weight=0)
        self._record_proposed_snapshot(state_type="rejected", tag=self.proposed.solver_tag, observations=self.proposed.observations)

    def _compute_log_posterior_terms(self, parameters: npt.NDArray, observation: npt.NDArray,
                                     solver_tag: int = 0) -> tuple[float, float]:
        """Returns ``(log_likelihood, log_prior)`` for one parameter/observation pair."""
        if solver_tag < 0:
            return -np.inf, -np.inf
        log_likelihood = self.likelihood.logpdf(observation)
        log_prior = self.prior.logpdf(parameters)
        return log_likelihood, log_prior

    def _draw_acceptance_decision(self, log_acceptance_probability: float) -> bool:
        temp = self._generator.uniform(0.0, 1.0)
        if np.log(temp) < log_acceptance_probability + np.log(self.stage.artificial_acceptance_multiplicator):
            return True  # accepted
        else:
            return False  # rejected

    def _record_current_sample(self) -> None:
        if self.conf.transform_before_saving:
            row: List[Any] = [1+self.counter_rejected_current] + list(self.prior.transform(self.current.parameters))
        else:
            row: List[Any] = [1+self.counter_rejected_current] + list(self.current.parameters)
        row.append(self.current.log_posterior)
        self.monitor(data_name="samples", row=row, condition=self.stage.save_to_file)

    def _send_to_collector(self, sample: Sample, weight: float) -> None:
        parameters = sample.parameters.copy()
        assert sample.observations is not None
        observations = sample.observations.copy()
        if self.conf.transform_before_surrogate:
            parameters = self.prior.transform(parameters)
        assert self.snapshot_collector is not None
        self.snapshot_collector.send_to_collector([parameters, observations, weight])

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
        if parameters1 is None:
            res = self.surrogate_evaluator(np.array(argument0))
            return res.ravel()
        else:
            res0 = self.surrogate_evaluator(np.array(argument0))
            res1 = self.surrogate_evaluator(np.array(argument1))
            return res0.ravel(), res1.ravel()

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
        evaluation = self.surrogate_evaluator(np.array([argument])).reshape(-1)
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


Algorithm_PARENT = AlgorithmBase


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
        counter_subchain_accepted = 0
        subchain_current = self.current.copy()
        correction_log_ratio = 0.0
        for _ in range(self.stage.subchain_max_length):
            bool_evaluator_changed = self._refresh_surrogate_evaluator_if_needed()
            subchain_proposed = self._propose_new_sample(subchain_current.parameters)
            self._evaluate_surrogate_transition( # evaluate surrogate for current, subchain_surrent, subchain_proposed
                subchain_current=subchain_current,
                subchain_proposed=subchain_proposed,
                surrogate_evaluator_changed=bool_evaluator_changed,
            )
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
        # TODO REMOVE THIS, just for debugging:
        """
        tmp_exact = evaluate_on_a_grid(self.observation_provider,par0_grid=np.linspace(-3, 3, 30), par1_grid=np.linspace(-3, 3, 30), filename=f"rank{self.rank_world}_exact_grid.png")
        tmp_surr  = evaluate_on_a_grid(self.surrogate_evaluator, par0_grid=np.linspace(-3, 3, 30), par1_grid=np.linspace(-3, 3, 30), filename=f"rank{self.rank_world}_initial_surrogate_grid.png")
        # save difference as image:
        import matplotlib.pyplot as plt
        plt.imshow(np.abs(tmp_exact - tmp_surr).reshape(len(np.linspace(-3, 3, 30)), len(np.linspace(-3, 3, 30))), extent=(-3, 3, -3, 3), origin='lower')
        plt.colorbar()
        plt.xlabel('par0')
        plt.ylabel('par1')
        plt.title('Exact - initial surrogate')
        plt.savefig(f"rank{self.rank_world}_exact_minus_initial_surrogate_grid.png")
        plt.close()"""
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
                #M row = [k] + [self.proposed.log_posterior] + [self.proposed.log_posterior_approx]
                assert self.proposed.log_likelihood_approx is not None
                assert self.current.log_likelihood_approx is not None
                """ TODO: Allow surrogate updates during subchain??
                likelihood_part, _ = self.proposal.get_log_acceptance_probability( # just for DEBUG, will be deleted
                    self.proposed.log_likelihood_approx,
                    self.current.log_likelihood_approx,
                    self.proposed.log_prior,
                    self.current.log_prior,
                )
                print(f"rank {self.rank_world} acceptance log-probability: approx {likelihood_part:.8f}, correction {correction_log_ratio:.8f}, counter subchain {counter_subchain}", flush=True)"""
                # 2 lines of weird hot fix instead of:
                # accepted = self._draw_acceptance_decision(log_acceptance_prob_exact - correction_log_ratio)
                exact_likelihood_log_ratio = self.proposed.log_likelihood - self.current.log_likelihood
                accepted = self._draw_acceptance_decision(exact_likelihood_log_ratio - correction_log_ratio)
                #M self.monitor(data_name="accepted" if accepted else "rejected", row=row, condition=self.stage.save_to_file)          )
                if accepted:
                    outer_accepted = True
                    self._handle_acceptance()
                else:
                    self._handle_rejection()
            else:  # proposed sample is the same as current sample, sample is automatically accepted, the chain remains here
                self._transition_to_prerejected()
                self._record_proposed_snapshot(state_type="prerejected", tag=0, observations=self.proposed.observations_approx)
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
        # TODO REMOVE THIS, just for debugging:
        """
        tmp_surr = evaluate_on_a_grid(self.surrogate_evaluator,par0_grid=np.linspace(-3, 3, 30), par1_grid=np.linspace(-3, 3, 30), filename=f"rank{self.rank_world}_final_surrogate_grid.png")
        # save difference as image:
        import matplotlib.pyplot as plt
        plt.imshow(np.abs(tmp_exact - tmp_surr).reshape(len(np.linspace(-3, 3, 30)), len(np.linspace(-3, 3, 30))), extent=(-3, 3, -3, 3), origin='lower')
        plt.colorbar()
        plt.xlabel('par0')
        plt.ylabel('par1')
        plt.title('Exact - initial surrogate')
        plt.savefig(f"rank{self.rank_world}_exact_minus_final_surrogate_grid.png")
        plt.close()
        """
        self._finalize_run()


