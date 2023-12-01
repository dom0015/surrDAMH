#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Prior:
    """
    Parent class for priors.
    The prior distribution is a function composition: transform(internal_prior()).

    """
    def __init__(self):
        self.mean = 0.0
        self.sd_approximation = None  # for LHS
        pass

    def transform(self, sample):
        """
        The sample is transformed before the solver is applied to it,
        and (optionally) before writing to a file.
        If not overridden, it remaint the identity.
        """
        return sample

    def calculate_log_prior(self, sample):
        """
        Calculates logarithm of prior pdf in given sample
        for the INTERNAL prior distribution
        (up to an additive constant).
        """
        return 0
