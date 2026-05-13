#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def evaluate_on_a_grid(Solver, par0_grid, par1_grid, filename):
    # create a grid of surrogate evaluations
    # only for no_parameters==2
    if Solver.no_parameters != 2:
        raise ValueError("evaluate_on_a_grid is only implemented for 2 parameters")

    par0_grid = par0_grid.reshape(-1)
    par1_grid = par1_grid.reshape(-1)
    par_grid = np.array(np.meshgrid(par0_grid, par1_grid)).T.reshape(-1, 2)
    # take just first output dimension for visualization, evaluate in a loop one by one
    obs_grid = np.zeros((len(par_grid), ))
    for i, par in enumerate(par_grid):
        obs_grid[i] = Solver.__call__(par)[0]

    # plot it as an image and save to a file
    import matplotlib.pyplot as plt
    plt.imshow(obs_grid.reshape(len(par0_grid), len(par1_grid)), extent=(par0_grid[0], par0_grid[-1], par1_grid[0], par1_grid[-1]), origin='lower')
    plt.colorbar()
    plt.xlabel('par0')
    plt.ylabel('par1')
    plt.title('Surrogate evaluation')
    plt.savefig(filename)
    plt.close()
    return obs_grid.reshape(len(par0_grid), len(par1_grid))