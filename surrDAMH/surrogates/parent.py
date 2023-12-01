#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy.typing as npt


class Evaluator:
    def __init__(self) -> None:
        pass

    def __call__(self, datapoints: npt.NDArray) -> npt.NDArray:
        """
        Evaluates the surrogate model in datapoints.

        datapoints shape: (number of datapoints, no_parameters)

        output NDArray shape: (number of datapoints, no_observations)
        """
        pass


class Updater:
    """
    Parent class for surrogate model updaters.
    """
    def __init__(self, no_parameters: int, no_observations: int) -> None:
        pass

    def delayed_init(self, data):
        """
        Additional settings of the surrogate model.
        """
        pass

    def add_data(self, parameters: npt.NDArray, observations: npt.NDArray, weights: npt.NDArray) -> None:
        """
        Adds more snapshots to the surrogate model.
        parameters shape: (number of snapshots, no_parameters)
        observations shape: (number of snapshots, no_observations)
        weights shape: (number of snapshots, 1)
        """
        pass

    def get_evaluator(self) -> Evaluator:
        """
        Returns Evaluator instance.
        """
        pass
