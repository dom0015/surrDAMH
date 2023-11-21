#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:19:41 2020

@author: simona
"""

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import time


def cov_function(r, sigma, lam):
    return np.power(sigma, 2)*np.exp(-r/lam)


def calculate_dist(nx, ny, lx, ly):
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)
    return np.sqrt(np.power(X, 2)+np.power(Y, 2))


def calculate_Cov(nx, ny, lx, ly, sigma, lam, show=False):
    distances = calculate_dist(nx, ny, lx, ly)
    cov_dist = cov_function(distances, sigma, lam)
    blocks = list(scipy.linalg.toeplitz(cov_dist[i, :]) for i in range(ny))
    structure = scipy.linalg.toeplitz(range(ny))
    blocks_list = [None] * ny
    for i in range(ny):
        blocks_list[i] = list(blocks[j] for j in structure[i, :])
    Cov = np.block(blocks_list)
    if show:
        plt.imshow(Cov)
        plt.show()
    return Cov


def calculate_Cholesky_factor(Cov):
    t = time.time()
    L = scipy.linalg.cholesky(Cov)
    print('Chol factorization time:', time.time() - t)
    return L


def sample_using_Cholesky(L, seed=None, show=True, nx=None):
    if seed is not None:
        np.random.seed(seed)
    nxny = L.shape[0]
    grf = np.matmul(np.transpose(L), np.random.randn(nxny))
    if show:
        if nx is None:
            nx = int(np.sqrt(nxny))
            ny = nx
        else:
            ny = int(nxny/nx)
        plt.imshow(grf.reshape((ny, nx)))
        plt.show()
    return grf


def calculate_eig(Cov):
    # t = time.time()
    D, V = np.linalg.eigh(Cov)
    # print('eigh factorization time:',time.time() - t)
    # sort eigenvalues and eigenvectors:
    indices = np.flip(np.argsort(D))
    Dsorted = D[indices]
    Vsorted = V[:, indices]
    return Dsorted, Vsorted


def eigenfunctions(V, indices, nx, ny, lx, ly, lam_new=None, lam_orig=None):
    if lam_new is None:
        x = np.linspace(0, lx, nx)
        y = np.linspace(0, ly, ny)
    else:
        coef = lam_orig/lam_new
        x = np.linspace(lx/2*(1-1/coef), lx/2*(1+1/coef), nx)  # cut form middle
        y = np.linspace(ly/2*(1-1/coef), ly/2*(1+1/coef), ny)  # cut from middle
        # x = np.linspace(0,lx/coef,nx) # cut from (0,0)
        # y = np.linspace(0,ly/coef,ny) # cut from (0,0)
    f = [None] * len(indices)
    for i, index in enumerate(indices):
        f[i] = scipy.interpolate.interp2d(x, y, V[:, index].reshape((ny, nx)), kind='cubic')
    return f


def evaluate_eigenfunctions(f, x, y, show=False, indices=None):
    n = len(f)
    z = list(f[i](x, y) for i in range(n))
    if show:
        plot_multiple_imshow(z, indices)
    return z


def evaluate_engenvalues(D, indices, sigma_new=None, sigma_orig=None):
    if sigma_new is None:
        return D[indices]
    else:
        return D[indices]/sigma_orig/sigma_orig*sigma_new*sigma_new


def plot_grf(eta, grf_path=None, cmap="viridis"):
    if grf_path is None:
        grf_path = 'unit30.pckl'
    grf_instance = GRF(grf_path, truncate=len(eta))
    grf_instance.plot_grf(eta, cmap)


def plot_eigenvectors(V, indices, nx=None):
    # indices ... columns of V to be visualized on the grid
    if nx is None:
        nx = int(np.sqrt(V.shape[0]))
        ny = nx
    else:
        ny = int(V.shape[0]/nx)
    IMG = list(V[:, i].reshape((ny, nx)) for i in indices)
    plot_multiple_imshow(IMG, indices)


def plot_eigenvalues(D, indices):
    plt.plot(D[indices])
    plt.show()


def plot_multiple_imshow(IMG, indices):
    if indices is None:
        indices = range(len(IMG))
    fig, axes = plt.subplots(1, len(indices), figsize=(12, 3))
    for i, index in enumerate(indices):
        axes[i].imshow(IMG[i])
        axes[i].set_xlabel("$i={0}$".format(index))
    plt.show()


def save_eig(filename, D, V, Cov, nx, ny, lx, ly, sigma, lam):
    import pickle
    f = open(filename, 'wb')
    pickle.dump([D, V, Cov, nx, ny, lx, ly, sigma, lam], f)
    f.close()


def load_eig(filename):
    import pickle
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def realization(D, V, nx, ny, lx, ly, seed=None, eta=None, truncate=None, show=False, lam_new=None, lam_orig=None):
    if truncate is None:
        truncate = len(D)
    D_new = D[:truncate].copy()
    V_new = V[:, :truncate].copy()
    np.random.seed(seed)
    if eta is None:
        eta = np.random.randn(truncate)
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    if lam_new is not None:
        eig_list = eigenfunctions(V_new, range(truncate), nx, ny, lx, ly, lam_new, lam_orig)
        for i in range(truncate):
            V_new[:, i] = eig_list[i](x, y).reshape((nx*ny,))
    M = np.matmul(V_new, eta*np.sqrt(D_new)).reshape((ny, nx))
    f = scipy.interpolate.interp2d(x, y, M, kind='cubic')
    if show:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(M)
        axes[0].set_xlabel("$seed={0},truncate={1}$".format(seed, truncate))
        x2 = np.linspace(0, lx, 2*nx)
        y2 = np.linspace(0, ly, 2*ny)
        z = f(x2, y2)
        axes[1].imshow(z)
        axes[1].set_xlabel('smooth')
        plt.show()
    print(D_new.shape, V_new.shape, truncate)
    return M, f


def generate_unit30():
    # rectangular domain
    # points are uniformly spaced on a rectangular grid
    nx = 30  # number of grid points in x direction
    ny = 30  # number of grid points in y direction
    lx = 1  # length of the domain in x direction
    ly = 1  # length of the domain in y direction
    sigma = 1  # std
    lam = 0.1  # autocorrelation length
    Cov = calculate_Cov(nx, ny, lx, ly, sigma, lam, show=False)
    D, V = calculate_eig(Cov)
    filename = 'unit30.pckl'
    save_eig(filename, D, V, Cov, nx, ny, lx, ly, sigma, lam)


def demo_generate_and_save():
    # rectangular domain
    # points are uniformly spaced on a rectangular grid
    nx = 50  # number of grid points in x direction
    ny = 50  # number of grid points in y direction
    lx = 10  # length of the domain in x direction
    ly = 10  # length of the domain in y direction
    sigma = 1  # std
    lam = 1  # autocorrelation length
    Cov = calculate_Cov(nx, ny, lx, ly, sigma, lam, show=False)
    D, V = calculate_eig(Cov)
    filename = 'demo.pckl'
    save_eig(filename, D, V, Cov, nx, ny, lx, ly, sigma, lam)
    indices = [0, 1, 3, 4, 5]
    plot_eigenvectors(V, indices, nx)


def demo_realization():
    filename = 'demo.pckl'
    D, V, Cov, nx, ny, lx, ly, sigma, lam = load_eig(filename)
    for i in [20, 100, None]:
        realization(D, V, nx, ny, lx, ly, seed=2, truncate=i, show=True)


class GRF:
    def __init__(self, filename, truncate=None, lam_new=None, sigma_new=None):
        D, V, Cov, self.nx, self.ny, self.lx, self.ly, sigma_orig, lam_orig = load_eig(filename)
        # self.nx, self.ny ... grid of the eigenfunctions (possibly coarse)
        self.x = np.linspace(0, self.lx, self.nx)
        self.y = np.linspace(0, self.ly, self.ny)
        if truncate is None:
            self.D = D.copy()
            self.V = V.copy()
        else:
            self.D = D[:truncate].copy()
            self.V = V[:, :truncate].copy()
        if sigma_new is None:
            self.sigma = sigma_orig
        else:
            self.D = self.D*sigma_new*sigma_new/sigma_orig/sigma_orig
            self.sigma = sigma_new
        if lam_new is None:
            self.lam = lam_orig
        else:
            eig_list = eigenfunctions(self.V, range(truncate), self.nx, self.ny, self.lx, self.ly, lam_new, lam_orig)
            for i in range(truncate):
                self.V[:, i] = eig_list[i](self.x, self.y).reshape((self.nx*self.ny,))
            self.lam = lam_new
        self.no_truncate = truncate

    def truncate(self, truncate):
        self.D = self.D[:truncate]
        self.V = self.V[:, :truncate]
        self.no_truncate = truncate

    def realization_grid_orig(self, eta):
        # V, D already has sigma_new and lam_new
        M = np.matmul(self.V, eta*np.sqrt(self.D)).reshape((self.ny, self.nx))
        return M

    def realization_as_function(self, eta):
        # V, D already has sigma_new and lam_new
        M = np.matmul(self.V, eta*np.sqrt(self.D)).reshape((self.ny, self.nx))
        f = scipy.interpolate.RectBivariateSpline(self.x, self.y, M)
        return f

    def realization_interfaces(self, eta, quantiles=[0.5, 1.0], nx_new=None, ny_new=None):
        f = self.realization_as_function(eta)
        if nx_new is None:
            nx_new = self.nx
        if ny_new is None:
            ny_new = self.ny
        x = np.linspace(0, self.lx, nx_new)
        y = np.linspace(0, self.ly, ny_new)
        R = f(x, y)
        bounds = np.quantile(R, quantiles)
        interfaces = 0*R
        for i in range(len(quantiles)-1):
            interfaces[R > bounds[i]] = i+1
        return interfaces

    def realization_interfaces_apply(self, eta, x, y, quantiles=[0.5, 1.0]):
        f = self.realization_as_function(eta)
        tmp = 0*x
        for i in range(len(x)):
            tmp[i] = f(x[i], y[i])
        bounds = np.quantile(tmp, quantiles)
        interfaces_applied = 0*x
        for i in range(len(quantiles)-1):
            interfaces_applied[tmp > bounds[i]] = i+1
        return interfaces_applied

    def plot_realization_interfaces(self, eta=None, quantiles=[0.5, 1.0], nx_new=None, ny_new=None, cmap="viridis"):
        if eta is None:
            eta = np.ones((self.no_truncate,))
        interfaces = self.realization_interfaces(eta, quantiles, nx_new, ny_new)
        fig, axes = plt.subplots(1, 1, figsize=(4, 3), sharey=True)
        axes.imshow(interfaces, origin="lower", extent=[0, self.lx, 0, self.ly], cmap=cmap)
        plt.show()

    def realization_grid_new(self, eta, x_new, y_new):
        # V, D already has sigma_new and lam_new
        M = np.matmul(self.V, eta*np.sqrt(self.D)).reshape((self.ny, self.nx))
        f = scipy.interpolate.RectBivariateSpline(self.x, self.y, M)
        M_new = f(x_new, y_new)
        return M_new

    def samples_mean_and_std(self, samples, x_new=None, y_new=None):
        temp = np.transpose(samples*np.sqrt(self.D))
        M = np.matmul(self.V, temp)
        M_mean = np.mean(M, axis=1)
        M_std = np.std(M, axis=1)
        if x_new is None:
            M_mean = M_mean.reshape((self.ny, self.nx))
            M_std = M_std.reshape((self.ny, self.nx))
        else:
            f_mean = scipy.interpolate.RectBivariateSpline(self.x, self.y, M_mean)
            f_std = scipy.interpolate.RectBivariateSpline(self.x, self.y, M_std)
            M_mean = f_mean(x_new, y_new)
            M_std = f_std(x_new, y_new)
        return M_mean, M_std

    def plot_grf(self, eta, cmap="viridis"):
        z = self.realization_grid_new(eta, np.linspace(0, self.lx, self.nx), np.linspace(0, self.ly, self.ny))
        fig, axes = plt.subplots(1, 1, figsize=(4, 3), sharey=True)
        axes.imshow(z, origin="lower", extent=[0, self.lx, 0, self.ly], cmap=cmap)
        plt.show()
