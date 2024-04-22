#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Likelihood:
    """
    Parent class for likelihoods.
    """
    def __init__(self):
        pass

    def logpdf(self, observations):
        """
        Calculates logarithm of the likelihood for given observations.
        """
        return 0
