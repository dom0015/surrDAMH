#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical post-processing of one sampling run: weighted moments, autocorrelation,
effective sample size, Gelman-Rubin R-hat, cost per uncorrelated sample and best-fit
selection from the raw snapshots.

Split out of the former single-file ``surrDAMH/post_processing.py`` (WS9b). Of the
package's own modules, this one imports only ``base`` -- so the private ``raw_data``
column helpers used by both ``find_best_fits`` here and the readers/plots live here as
well.

``SamplesStatistics(SamplesBase)`` is the first link of the mixin chain that is combined
into :class:`surrDAMH.post_processing.Samples` in ``loading.py``
(``SamplesStatistics`` -> ``SamplesPlots`` -> ``SamplesReports`` -> ``Samples``); its
methods use the attributes and resolver helpers (``list_of_stages``, ``stage_names``,
``no_parameters``, ``summary``, ``_resolve_stages``, ...) declared on ``SamplesBase``.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Literal, TYPE_CHECKING

import emcee
import numpy as np
import pandas as pd

from surrDAMH.modules.manifest import RunFormatError
from surrDAMH.post_processing.base import SamplesBase

if TYPE_CHECKING:  # pragma: no cover - typing only
    from surrDAMH.post_processing.loading import Samples


def _raw_data_parameter_columns(no_parameters: int) -> List[str]:
    return [f"par_{i}" for i in range(no_parameters)]


def _raw_data_observation_columns(chosen_observations) -> List[str]:
    return [f"obs_{int(i)}" for i in np.atleast_1d(chosen_observations)]


def _select_columns(frame: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """``frame[columns]`` as a float array; a column the frame does not have becomes NaN.

    This is the permissive accessor, used after the caller has checked that the columns it
    actually needs are present (see :func:`_require_raw_data_columns`). ``read_run`` is the
    strict reader -- it verifies the full v2 header of every file it loads.
    """
    values = np.full((len(frame), len(columns)), np.nan, dtype=float)
    for position, name in enumerate(columns):
        if name in frame.columns:
            values[:, position] = np.asarray(frame[name], dtype=float)
    return values


def _require_raw_data_columns(frame: pd.DataFrame, path: str, columns: List[str],
                              ranking_mode: str) -> None:
    """
    Refuse a ``raw_data`` chunk that lacks a column the requested ranking needs.

    Finding P10: these columns used to be read permissively, so a truncated or
    foreign-format file produced all-NaN scores, which then became ``-inf`` and ranked the
    rows last -- an arbitrary "best fit" table that looked complete. WS9b makes it an error.
    """
    missing = [name for name in columns if name not in frame.columns]
    if not missing:
        return
    raise RunFormatError(
        f"{path}: raw_data is missing the column(s) {missing} required by "
        f"ranking_mode={ranking_mode!r}.\n"
        f"  found: {list(frame.columns)}\n"
        f"Output format v2 writes them for every run (docs/outputs.md); a file without them "
        f"was not written by this version and there is no converter (decision 6)."
    )


def _rank_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    """Best-to-worst order of ``scores``, delegated to :func:`rank_best_fit_candidates`."""
    if mode == "l2":
        return rank_best_fit_candidates(misfits=scores, mode="l2")
    if mode == "posterior":
        return rank_best_fit_candidates(posteriors=scores, mode="posterior")
    return rank_best_fit_candidates(likelihoods=scores, mode="likelihood")


def rank_best_fit_candidates(misfits: np.ndarray | None = None,
                             posteriors: np.ndarray | None = None,
                             likelihoods: np.ndarray | None = None,
                             mode: str = "l2") -> np.ndarray:
    """Return indices that rank candidates from best to worst for a requested metric."""
    normalized_mode = mode.lower()
    if normalized_mode == "l2":
        if misfits is None:
            raise ValueError("misfits must be provided when ranking_mode='l2'")
        values = np.asarray(misfits, dtype=float)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            raise ValueError("No finite L2 misfit values available for ranking")
        candidate_idx = np.flatnonzero(finite_mask)
        order = np.argsort(values[finite_mask], kind="mergesort")
        return candidate_idx[order]

    if normalized_mode == "posterior":
        if posteriors is None:
            raise ValueError("posteriors must be provided when ranking_mode='posterior'")
        values = np.asarray(posteriors, dtype=float)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            raise ValueError("No finite posterior values available for ranking")
        candidate_idx = np.flatnonzero(finite_mask)
        order = np.argsort(-values[finite_mask], kind="mergesort")
        return candidate_idx[order]

    if normalized_mode == "likelihood":
        if likelihoods is None:
            raise ValueError("likelihoods must be provided when ranking_mode='likelihood'")
        values = np.asarray(likelihoods, dtype=float)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            raise ValueError("No finite likelihood values available for ranking")
        candidate_idx = np.flatnonzero(finite_mask)
        order = np.argsort(-values[finite_mask], kind="mergesort")
        return candidate_idx[order]

    raise ValueError("Unsupported best-fit ranking mode. Expected 'l2', 'posterior', or 'likelihood'.")


class SamplesStatistics(SamplesBase):
    """Statistics layer of :class:`surrDAMH.post_processing.Samples` (see module docstring)."""

    def get_mean_and_cov(self, stages_to_disp: Iterable | None = None, burn_in: List[List[int]] | None = None,
                         npy_filepath: str | None = None, chains_to_disp: Iterable | None = None):
        """
        Multiplicity-weighted posterior mean and covariance of the compressed samples.

        Args:
            stages_to_disp: stage indices to include (None = all stages).
            burn_in: leading compressed rows to drop, per displayed stage and per chain.
            npy_filepath: if given, ``[mean, cov]`` is also saved there as ``.npy``.
            chains_to_disp: chain indices to include (None = every chain of each stage).
        """
        stages_to_disp = self._resolve_stages(stages_to_disp)
        burn_in = self._burn_in_for(stages_to_disp, burn_in)
        all_x = np.zeros((0, self.no_parameters))
        all_w = np.zeros((0, ))
        for idj, j in enumerate(stages_to_disp):
            for s in self._resolve_chains(j, chains_to_disp):
                tmp_x = self.list_of_stages[j].samples_compressed[s][burn_in[idj][s]:, :]
                tmp_w = self.list_of_stages[j].weights[s][burn_in[idj][s]:]
                all_x = np.concatenate((all_x, tmp_x))
                all_w = np.concatenate((all_w, tmp_w))
        mean = np.average(all_x, axis=0, weights=all_w)
        cov = np.cov(all_x, rowvar=False, fweights=all_w)
        if npy_filepath:
            # save to .npy file:
            np.save(npy_filepath, [mean, cov])
            print("Mean and covariance saved to", npy_filepath, flush=True)
        return mean, cov

    def calculate_CpUS(self, list_of_stages_groups: List[List[int]], surrogate_cost_ratio: float = 0.0,
                       chains_to_disp: Iterable | None = None):
        """
        Calculates the cost per uncorrelated sample (CpUS)
        for all groups of stages. Returns extended summary.

        Args:
            list_of_stages_groups: groups of stage indices; each group gets one CpUS value.
            surrogate_cost_ratio: cost of one surrogate evaluation relative to an exact one.
            chains_to_disp: chain indices the autocorrelation is estimated from
                (None = every chain). The ``accepted``/``rejected``/``sum`` counters the
                ratio comes from are whole-stage totals either way.
        """
        ratio_evaluated = (self.summary["accepted"] + self.summary["rejected"]) / self.summary["sum"]
        # add column to summary:
        self.summary["ratio_eval"] = ratio_evaluated
        autocorr_stages = np.zeros((self.no_stages,))
        cpus_stages = -np.ones((self.no_stages,), dtype=float)
        for stages_to_disp in list_of_stages_groups:
            print("Stages:", stages_to_disp)
            try:
                autocorr = Autocorrelation(self, stages_to_disp=stages_to_disp,
                                           chains_to_disp=chains_to_disp)
                autocorr.calculate_autocorr_function()
                a, _ = autocorr.calculate_autocorr_time_mean()
                a = np.mean(a)
                autocorr_stages[stages_to_disp] = a
                ratio_loc = self.summary["ratio_eval"].iloc[stages_to_disp]
                cpus_stages[stages_to_disp] = a * (ratio_loc + surrogate_cost_ratio)
            except Exception as e:
                print(f"CpUS unavailable for stages {stages_to_disp}: {e}")
        # add columns to summary:
        self.summary["autocorr"] = autocorr_stages
        self.summary["CpUS"] = cpus_stages
        return self.summary

    def calculate_effective_sample_size(self, stages_to_disp: Iterable | None = None, burn_in: List[List[int]] | None = None,
                                        chains_to_disp: Iterable | None = None):
        """
        Calculate effective sample size (ESS) for each parameter.
        ESS = total_samples / autocorrelation_time

        Args:
            stages_to_disp (list of int): stages to include (None = all)
            burn_in (list of list of int): burn-in per stage/chain
            chains_to_disp (list of int): chains to include (None = every chain)

        Returns:
            dict with ESS per parameter and overall statistics, or None if the
            autocorrelation time could not be estimated (e.g. chains too short).
        """
        stages_to_disp = self._resolve_stages(stages_to_disp)
        burn_in = self._burn_in_for(stages_to_disp, burn_in)

        try:
            autocorr = Autocorrelation(self, stages_to_disp=stages_to_disp, burn_in=burn_in,
                                       chains_to_disp=chains_to_disp)
            autocorr.calculate_autocorr_function()
            autocorr_times, _ = autocorr.calculate_autocorr_time_mean()

            # Calculate total samples
            total_samples = 0
            for idj, j in enumerate(stages_to_disp):
                for s in self._resolve_chains(j, chains_to_disp):
                    total_samples += len(self._decompressed_samples(j)[s][burn_in[idj][s]:, 0])

            ess_per_param = {i: total_samples / float(autocorr_times[i]) for i in range(self.no_parameters)}
            ess_overall = np.mean(list(ess_per_param.values()))
            
            return {
                'ess_per_param': ess_per_param,
                'ess_overall': ess_overall,
                'total_samples': total_samples,
                'autocorr_times': autocorr_times
            }
        except Exception as e:
            print(f"Error calculating ESS: {e}")
            return None

    def _collect_samples_matrix(self, stages_to_disp: Iterable | None = None,
                                burn_in: List[List[int]] | None = None,
                                chains_to_disp: Iterable | None = None):
        """All decompressed samples of the selected stages/chains, stacked row-wise."""
        stages_to_disp = self._resolve_stages(stages_to_disp)
        burn_in = self._burn_in_for(stages_to_disp, burn_in)

        all_x = np.zeros((0, self.no_parameters))
        for idj, j in enumerate(stages_to_disp):
            for s in self._resolve_chains(j, chains_to_disp):
                all_x = np.concatenate((all_x, self._decompressed_samples(j)[s][burn_in[idj][s]:, :]))
        return all_x

    def calculate_gelman_rubin(self, stages_to_disp: Iterable | None = None,
                               burn_in: List[List[int]] | None = None,
                               chains_to_disp: Iterable | None = None):
        """
        Calculate Gelman-Rubin potential scale reduction factor (R-hat)
        for each parameter using parallel chains.

        Args:
            stages_to_disp (list of int): stages to include (None = all)
            burn_in (list of list of int): burn-in per stage/chain
            chains_to_disp (list of int): chains to include (None = every chain)

        Returns:
            dict with per-parameter and aggregate R-hat statistics, or None if fewer than
            two chains with at least two samples each are available.
        """
        stages_to_disp = self._resolve_stages(stages_to_disp)
        burn_in = self._burn_in_for(stages_to_disp, burn_in)

        chain_data = []
        for chain_idx in self._resolve_chains(stages_to_disp[0], chains_to_disp):
            chain_samples = np.zeros((0, self.no_parameters))
            for idj, stage_idx in enumerate(stages_to_disp):
                if chain_idx >= self.list_of_stages[stage_idx].no_chains:
                    # a stage may have been written by fewer chains than the first one
                    continue
                tmp = self._decompressed_samples(stage_idx)[chain_idx][burn_in[idj][chain_idx]:, :]
                chain_samples = np.concatenate((chain_samples, tmp))
            if chain_samples.shape[0] >= 2:
                chain_data.append(chain_samples)

        if len(chain_data) < 2:
            return None

        n = min(chain.shape[0] for chain in chain_data)
        if n < 2:
            return None

        samples = np.stack([chain[:n, :] for chain in chain_data], axis=0)
        m = samples.shape[0]

        chain_means = np.mean(samples, axis=1)
        grand_mean = np.mean(chain_means, axis=0)
        B = (n / (m - 1)) * np.sum((chain_means - grand_mean) ** 2, axis=0)

        chain_vars = np.var(samples, axis=1, ddof=1)
        W = np.mean(chain_vars, axis=0)
        W_safe = np.maximum(W, 1e-15)

        var_hat = ((n - 1) / n) * W + B / n
        rhat = np.sqrt(var_hat / W_safe)

        return {
            "rhat_per_param": {i: float(rhat[i]) for i in range(self.no_parameters)},
            "rhat_max": float(np.max(rhat)),
            "rhat_mean": float(np.mean(rhat)),
            "n_per_chain": int(n),
            "m_chains": int(m)
        }

    def _load_snapshot_parameters_and_observations(self, no_observations: int,
                                                   chains_to_disp: Iterable | None = None,
                                                   stages_to_disp: Iterable | None = None):
        if no_observations <= 0:
            return np.empty((0, self.no_parameters)), np.empty((0, 0))

        requested_chains = None if chains_to_disp is None else [int(c) for c in chains_to_disp]
        stage_indices = self._resolve_stages(stages_to_disp)

        par_blocks = []
        obs_blocks = []
        parameter_columns = _raw_data_parameter_columns(self.no_parameters)
        observation_columns = _raw_data_observation_columns(np.arange(no_observations))

        for stage_idx in stage_indices:
            stage_name = self.stage_names[stage_idx]
            dirname = os.path.join(self.sampling_output_dir, "raw_data", stage_name)
            if not os.path.isdir(dirname):
                continue

            files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
            files.sort()
            for chain_idx in (range(len(files)) if requested_chains is None else requested_chains):
                if chain_idx >= len(files):
                    continue
                path_samples = os.path.join(dirname, files[chain_idx])
                try:
                    df_samples = pd.read_csv(path_samples)
                except pd.errors.EmptyDataError:
                    continue

                mask = df_samples["state_type"].astype(str) != "prerejected"
                if not mask.any():
                    continue

                filtered = df_samples.loc[mask]
                par_block = _select_columns(filtered, parameter_columns)
                obs_block = _select_columns(filtered, observation_columns)
                if par_block.size == 0 or obs_block.size == 0:
                    continue
                par_blocks.append(par_block)
                obs_blocks.append(obs_block)

        if not par_blocks:
            return np.empty((0, self.no_parameters)), np.empty((0, no_observations))

        return np.vstack(par_blocks), np.vstack(obs_blocks)

    def find_best_fits(self, no_observations: int, observations: np.ndarray,
                       n_best: int = 10, chains_to_disp: Iterable | None = None,
                       stages_to_disp: Iterable | None = None,
                       par_names: List[str] | None = None,
                       ranking_mode: Literal["l2", "posterior", "likelihood"] = "l2"):
        """
        Find parameter snapshots with the best score according to the requested ranking mode.

        Supported modes:
            - "l2": smallest L2 misfit to the supplied observations
            - "posterior": largest log-posterior value
            - "likelihood": largest log-likelihood value

        Only the exactly evaluated snapshots take part: ``prerejected`` rows have no exact
        observations (their ``obs_*`` block is NaN in format v2) and are dropped. Rows whose
        score is not finite are dropped as well -- the ordering itself is delegated to
        :func:`rank_best_fit_candidates`, the unit-tested ranking function (finding P3).

        Args:
            no_observations: length of the observation vector.
            observations: reference observations (only used by ``ranking_mode="l2"``).
            n_best: number of snapshots to keep.
            chains_to_disp: chain indices to include (None = every chain).
            stages_to_disp: stage indices to include (None = all stages).
            par_names: column names for the parameters in the returned table.
            ranking_mode: "l2", "posterior" or "likelihood".

        Returns:
            tuple[DataFrame, ndarray, ndarray, ndarray]: table, parameters, model outputs, scores.

        Raises:
            ValueError: ``no_observations <= 0``, ``observations`` of the wrong length, or an
                unsupported ``ranking_mode``.
            RunFormatError: a ``raw_data`` file does not contain the columns this ranking
                mode needs (finding P10: this used to degrade silently to ``-inf`` scores,
                i.e. an arbitrary "best" table).
        """
        if no_observations <= 0:
            raise ValueError("no_observations must be positive.")

        observations = np.asarray(observations, dtype=float).reshape(-1)
        if observations.size != no_observations:
            raise ValueError(
                f"observations has length {observations.size}, expected {no_observations}."
            )

        n_keep = max(int(n_best), 0)
        if n_keep == 0:
            columns = ["rank", "l2_misfit"]
            param_names = par_names or [f"par_{i}" for i in range(self.no_parameters)]
            return (
                pd.DataFrame(columns=columns + param_names),
                np.empty((0, self.no_parameters)),
                np.empty((0, no_observations)),
                np.empty((0,)),
            )

        # chains are addressed by their raw_data FILE here, so "all chains" means "every
        # file this stage wrote" -- not "as many as global stage 0 has" (same class of bug
        # as finding P7).
        requested_chains = None if chains_to_disp is None else [int(c) for c in chains_to_disp]
        stage_indices = self._resolve_stages(stages_to_disp)

        best_par = np.empty((0, self.no_parameters), dtype=float)
        best_obs = np.empty((0, no_observations), dtype=float)
        best_scores = np.empty((0,), dtype=float)
        parameter_columns = _raw_data_parameter_columns(self.no_parameters)
        observation_columns = _raw_data_observation_columns(np.arange(no_observations))
        obs_reference = observations.reshape((1, -1))
        chunk_size = 50000
        normalized_mode = ranking_mode.lower()
        if normalized_mode not in {"l2", "posterior", "likelihood"}:
            raise ValueError("ranking_mode must be one of: 'l2', 'posterior', 'likelihood'")
        required_columns = ["state_type"] + parameter_columns + {
            "l2": observation_columns,
            "posterior": ["log_likelihood", "log_prior"],
            "likelihood": ["log_likelihood"],
        }[normalized_mode]

        for stage_idx in stage_indices:
            stage_name = self.stage_names[stage_idx]
            dirname = os.path.join(self.sampling_output_dir, "raw_data", stage_name)
            if not os.path.isdir(dirname):
                continue

            files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
            files.sort()
            for chain_idx in (range(len(files)) if requested_chains is None else requested_chains):
                if chain_idx >= len(files):
                    continue
                path_samples = os.path.join(dirname, files[chain_idx])
                try:
                    chunk_iter = pd.read_csv(path_samples, chunksize=chunk_size)
                except pd.errors.EmptyDataError:
                    continue

                for df_samples in chunk_iter:
                    _require_raw_data_columns(df_samples, path_samples, required_columns,
                                              normalized_mode)
                    mask = df_samples["state_type"].astype(str) != "prerejected"
                    if not mask.any():
                        continue

                    filtered = df_samples.loc[mask]
                    par_block = _select_columns(filtered, parameter_columns)
                    obs_block = _select_columns(filtered, observation_columns)
                    if par_block.size == 0 or obs_block.size == 0:
                        continue

                    if normalized_mode == "l2":
                        residuals = obs_block - obs_reference
                        score_block = np.sqrt(np.einsum("ij,ij->i", residuals, residuals))
                    elif normalized_mode == "posterior":
                        score_values = _select_columns(filtered, ["log_likelihood", "log_prior"])
                        score_block = score_values[:, 0] + score_values[:, 1]
                    else:
                        score_block = _select_columns(filtered, ["log_likelihood"])[:, 0]

                    # a NaN/inf score is not a "worst" candidate, it is no candidate at all
                    finite = np.isfinite(score_block)
                    if not finite.any():
                        continue
                    par_block = par_block[finite]
                    obs_block = obs_block[finite]
                    score_block = score_block[finite]

                    candidate_par = np.vstack((best_par, par_block))
                    candidate_obs = np.vstack((best_obs, obs_block))
                    candidate_scores = np.concatenate((best_scores, score_block))

                    keep_idx = _rank_scores(candidate_scores, normalized_mode)[:n_keep]
                    best_par = candidate_par[keep_idx]
                    best_obs = candidate_obs[keep_idx]
                    best_scores = candidate_scores[keep_idx]

        if best_scores.size == 0:
            columns = ["rank", "l2_misfit"]
            param_names = par_names or [f"par_{i}" for i in range(self.no_parameters)]
            return (
                pd.DataFrame(columns=columns + param_names),
                np.empty((0, self.no_parameters)),
                np.empty((0, no_observations)),
                np.empty((0,)),
            )

        param_names = par_names or [f"par_{i}" for i in range(self.no_parameters)]
        score_column = "l2_misfit"
        if normalized_mode == "posterior":
            score_column = "log_posterior"
        elif normalized_mode == "likelihood":
            score_column = "log_likelihood"
        data = {
            "rank": np.arange(1, len(best_scores) + 1, dtype=int),
            score_column: best_scores,
        }
        for param_idx in range(self.no_parameters):
            if param_idx < len(param_names):
                column_name = param_names[param_idx]
            else:
                column_name = f"par_{param_idx}"
            data[column_name] = best_par[:, param_idx]

        return (
            pd.DataFrame(data),
            best_par,
            best_obs,
            best_scores,
        )

    def compute_posterior_field_statistics(self, field_builder,
                                           stages_to_disp: Iterable | None = None,
                                           burn_in: List[List[int]] | None = None,
                                           n_max_samples: int | None = None,
                                           random_seed: int = 42,
                                           chains_to_disp: Iterable | None = None):
        """
        Compute posterior mean and standard deviation of a derived field.
        """
        all_x = self._collect_samples_matrix(stages_to_disp=stages_to_disp, burn_in=burn_in,
                                             chains_to_disp=chains_to_disp)
        if all_x.shape[0] == 0:
            raise ValueError("No posterior samples available for field statistics.")

        if n_max_samples is not None and n_max_samples > 0 and all_x.shape[0] > n_max_samples:
            rng = np.random.default_rng(random_seed)
            idx = rng.choice(all_x.shape[0], size=n_max_samples, replace=False)
            all_x = all_x[idx]

        first_field = np.asarray(field_builder(all_x[0]), dtype=float)
        field_sum = np.zeros_like(first_field, dtype=float)
        field_sum_sq = np.zeros_like(first_field, dtype=float)

        for sample in all_x:
            field = np.asarray(field_builder(sample), dtype=float)
            if field.shape != first_field.shape:
                raise ValueError("field_builder returned inconsistent field shapes.")
            field_sum += field
            field_sum_sq += field ** 2

        mean_field = field_sum / all_x.shape[0]
        var_field = np.maximum(field_sum_sq / all_x.shape[0] - mean_field ** 2, 0.0)
        std_field = np.sqrt(var_field)
        return mean_field, std_field


# AUTOCORRELATION:
# Autocorrelation analysis using emcee, Foreman-Mackey,
# adapted for the needs of the DAMH-SMU framework.
# Considers "N = no_chains" chains of different lengths (l_1, ..., l_N),
# each of the chains has "n = no_parameters" components.
# The samples in one chain form a numpy array of shape (l_i, n).
# All samples form a python list of length N.

class Autocorrelation:
    def __init__(self, samples: Samples, stages_to_disp: Iterable,
                 burn_in: List[List[int]] | None = None,
                 chains_to_disp: Iterable | None = None):
        """
        Autocorrelation analysis for a subset of stages.
        Using emcee (Foreman-Mackey).

        Args:
            samples (Samples): Samples object.
            stages_to_disp (list of int of length S): Stages to analyse. If None, all stages.
            burn_in (list of list of int): Burn-in for each analysed stage and each analysed
                chain. If None, set to [[0] * no_chains] * S.
            chains_to_disp (list of int): Chains to analyse. If None, every chain of the
                FIRST ANALYSED stage (finding P7: the chain count used to be taken from
                global stage 0, whether or not stage 0 was among the stages analysed).
        """
        self.samples = samples
        self.stages_to_disp = samples._resolve_stages(stages_to_disp)
        self.stages = [samples.list_of_stages[i] for i in self.stages_to_disp]
        # P7: the chains come from the stages actually analysed, not from global stage 0.
        self.chains_to_disp = samples._resolve_chains(self.stages_to_disp[0], chains_to_disp)
        self.no_chains = len(self.chains_to_disp)
        # burn_in is indexed [position in stages_to_disp][chain index within that stage],
        # the same convention as Samples._burn_in_for, so a chain subset stays consistent.
        self.burn_in = samples._burn_in_for(self.stages_to_disp, burn_in)

        self.samples_all_stages = []
        for _ in range(self.no_chains):
            self.samples_all_stages.append(np.zeros((0, self.samples.no_parameters)))
        for idx_stage, stage in enumerate(self.stages):
            begin = self.burn_in[idx_stage]
            end = stage.length
            stage_samples = samples._decompressed_samples(self.stages_to_disp[idx_stage])

            for idx, i in enumerate(self.chains_to_disp):
                x = stage_samples[i][begin[i]:end[i], :]
                self.samples_all_stages[idx] = np.concatenate((self.samples_all_stages[idx], x))

    def calculate_autocorr_function(self):
        self.autocorr_function = [None] * self.no_chains

        for i in range(self.no_chains):
            self.autocorr_function[i] = np.zeros(self.samples_all_stages[i].shape)
            for j in range(self.samples.no_parameters):
                self.autocorr_function[i][:, j] = emcee.autocorr.function_1d(self.samples_all_stages[i][:, j])
        self.length = [x.shape[0] for x in self.autocorr_function]
        print("Autocorr. functions calculated, shapes:", [i.shape for i in self.autocorr_function])

    def calculate_autocorr_function_mean(self):
        chains_range = range(self.no_chains)
        max_length = max(self.length)
        autocorr_function_mean = np.zeros((max_length, self.samples.no_parameters))
        for j in range(self.samples.no_parameters):
            tmp = np.zeros(max_length)
            count = np.zeros(max_length)
            for idx, i in enumerate(chains_range):
                tmp[:self.length[idx]] += self.autocorr_function[i][:, j]
                count[:self.length[idx]] += 1
            autocorr_function_mean[:, j] = tmp / count

        return autocorr_function_mean

    def calculate_autocorr_time(self, c=5, tol=50, quiet=True):
        autocorr_time = [None] * self.no_chains
        for i in range(self.no_chains):
            tmp = np.zeros((self.samples.no_parameters))
            for j in range(self.samples.no_parameters):
                tmp[j] = emcee.autocorr.integrated_time(self.samples_all_stages[i][:, j], c=c, tol=tol, quiet=quiet)
            autocorr_time[i] = tmp
        return autocorr_time

    def calculate_autocorr_time_mean(self, c: int = 5):
        autocorr_time_mean = [None] * self.samples.no_parameters
        autocorr_time_mean_beta = [None] * self.samples.no_parameters
        length = min(self.length)
        autocorr_function_mean = self.calculate_autocorr_function_mean()
        for j in range(self.samples.no_parameters):
            f = autocorr_function_mean[:length, j]
            autocorr_time_mean[j] = autocorr_FM(f, c)
            f = autocorr_function_mean[:, j]
            autocorr_time_mean_beta[j] = autocorr_FM(f, c)
        return autocorr_time_mean, autocorr_time_mean_beta


def auto_window(taus, c):
    # Automated windowing procedure following Sokal (1989)
    # from https://dfm.io/posts/autocorr/ Foreman-Mackey
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def autocorr_FM(f, c: int = 5):
    # from https://dfm.io/posts/autocorr/ Foreman-Mackey
    # first calculates all autocorr. functions, than averages them
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def decompress(samples_compressed, weights):
    # samples_compressed ... "compressed" samples from DAMH-SMU
    # weights ... counts of consecutive identical samples
    sum_w = np.sum(weights)
    cumsum_w = np.append(0, np.cumsum(weights))
    no_unique, no_parameters = samples_compressed.shape
    samples_decompressed = np.zeros((sum_w, no_parameters))
    for i in range(no_unique):
        samples_decompressed[cumsum_w[i]:cumsum_w[i + 1], :] = samples_compressed[i, :]
    return samples_decompressed
