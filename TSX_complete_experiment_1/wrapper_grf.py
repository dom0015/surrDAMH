"""
Wrapper for GRF-based TSX poroelastic model — Experiment 1 (45 parameters).

Parameter layout (45 parameters total):
  par[ 0:20]   KL coefficients xi_i for log10(K_h)   — hydraulic conductivity GRF
  par[20:40]   KL coefficients xi_i for log10(n)      — porosity GRF
  par[40]      K_d inner (drained bulk modulus) [Pa]   — zone-dependent
  par[41]      K_d outer (drained bulk modulus) [Pa]   — zone-dependent
  par[42]      G inner   (shear modulus)        [Pa]   — zone-dependent
  par[43]      G outer   (shear modulus)        [Pa]   — zone-dependent
  par[44]      K_s       (grain bulk modulus)    [Pa]   — scalar (uniform)

Spatially varying fields:
  - Hydraulic conductivity: K_h(x) = 10^(GRF_Kh(x) - 13),  k(x) = K_h / (rho_f * g)
  - Porosity:               n(x)   = 10^(GRF_n(x) - 2.5)
  - K_d(x), G(x):          sigmoid transition from inner to outer value
                            based on elliptical distance from tunnel

Derived quantities (spatially varying DG-0 fields):
  E_d(x)   = 9 * K_d(x) * G(x) / (3*K_d(x) + G(x))
  nu_d(x)  = (3*K_d(x) - 2*G(x)) / (2*(3*K_d(x) + G(x)))
  alpha(x) = 1 - K_d(x) / K_s
  S(x)     = (alpha(x) - n(x)) / K_s + n(x) / K_f    (K_f = 2.15e9 fixed)
  mu(x)    = G(x)
  lmbda(x) = E_d(x) * nu_d(x) / ((1+nu_d(x))*(1-2*nu_d(x)))

In-situ stresses sigma_x = 45 MPa, sigma_y = 12.8 MPa are fixed (not parameters).
"""

import os
import numpy as np
import dolfinx as dfx

from tunnel_grf import (
    load_mesh_and_domain_tags,
    load_kl_data,
    build_permeability_field,
    build_sigmoid_field_dg0,
    tsx_setup_and_computation_v2,
    uniform_time_steps,
    piecewise_uniform_time_steps,
    geometric_time_steps,
    SEC_IN_DAY,
)

# ---------------------------------------------------------------------------
# Default observation schedule (every 20 days from day 17 to 357)
# ---------------------------------------------------------------------------
DEFAULT_OBSERVATION_TIMES_DAYS = np.arange(17, 358, 20)

# Pa → metres of water head
PA_TO_MWH = 9806.0

# ---------------------------------------------------------------------------
# Fixed physical constants
# ---------------------------------------------------------------------------
RHO_F = 1000.0          # Fluid density [kg/m³]
MU_F  = 1.0e-3          # Fluid dynamic viscosity [Pa·s]
G_ACC = 9.81            # Gravitational acceleration [m/s²]
K_F   = 2.15e9           # Fluid bulk modulus [Pa] (fixed)

# Fixed in-situ stresses
SIGMA_X = 45e6           # Horizontal stress [Pa]
SIGMA_Y = 12.8e6         # Vertical stress [Pa]

# Number of KL modes used for each GRF (out of 100 precomputed)
N_KL = 20

# ---------------------------------------------------------------------------
# Reference observations (from Chandler 2002 field data)
# ---------------------------------------------------------------------------
observations = np.array([
    754.64805252, 755.01945387, 655.95978987, 594.39699416,
    530.43745741, 515.8816575,  494.97253552, 475.59152216,
    452.86384229, 433.26618307, 382.97842896, 351.99284788,
    332.17098259, 309.92537921, 300.5676362,  303.92681608,
    287.90724713, 291.92276435, 515.43370038, 587.81872627,
    600.23143393, 604.48910264, 600.39683158, 599.11176231,
    603.23124273, 599.23918313, 599.03415368, 601.54081621,
    584.89274096, 571.77321889, 559.71106412, 551.93560232,
    550.11891225, 547.36679965, 546.21690982, 542.02776456,
    182.31709971, 199.55739055, 212.15676692, 224.15477335,
    229.70971358, 238.39284514, 253.75478535, 262.23129018,
    268.99356638, 276.3197908,  278.0979942,  281.01668497,
    280.27204155, 284.22676206, 286.21291704, 290.6920895,
    294.08053105, 294.73164677,  48.08262448,  42.93917678,
     49.94181195,  61.35265637,  59.6642977,   75.47073607,
     81.53250879,  90.51661055,  90.82438687,  89.70334219,
     79.27229886,  79.20607144,  83.22591843,  77.02257697,
     78.10138551,  84.91265489,  75.86837226,  82.69544488,
])


class SolverTSX_grf:
    """Poroelastic TSX solver — Experiment 1 (45 parameters).

    Two GRF fields (K_h, porosity) via truncated KL expansion,
    zone-dependent K_d and G via sigmoid transition, scalar K_s.

    Parameters
    ----------
    n_kl_modes : int
        Number of KL modes to use per GRF field. Default 20.
    mesh_prefix : str or None
        Path prefix for mesh XDMF files (without extension).
        Defaults to files in the same directory as this script.
    kl_Kh_filepath : str or None
        HDF5 file with K_h KL eigenvalues/eigenvectors.
    kl_por_filepath : str or None
        HDF5 file with porosity KL eigenvalues/eigenvectors.
    time_steps : array-like or None
        Custom time step schedule in seconds.
    """

    def __init__(self, n_kl_modes=N_KL,
                 mesh_prefix=None,
                 kl_Kh_filepath=None,
                 kl_por_filepath=None,
                 time_steps=None):
        # Resolve default data paths relative to this file's location
        _data_dir = os.path.dirname(os.path.abspath(__file__))
        if mesh_prefix is None:
            mesh_prefix = os.path.join(_data_dir, "tsx_ellipses_very_coarse")
        if kl_Kh_filepath is None:
            kl_Kh_filepath = os.path.join(_data_dir,
                                          "tsx_ellipses_very_coarse_Kh_KL100.h5")
        if kl_por_filepath is None:
            kl_por_filepath = os.path.join(_data_dir,
                                           "tsx_ellipses_very_coarse_por_KL100.h5")

        # Load mesh
        self.mesh, self.cell_tags, _ = load_mesh_and_domain_tags(mesh_prefix)

        # Load KL data for hydraulic conductivity
        self.eigenvalues_Kh, self.eigenvectors_Kh = load_kl_data(kl_Kh_filepath)

        # Load KL data for porosity
        self.eigenvalues_por, self.eigenvectors_por = load_kl_data(kl_por_filepath)

        self.n_kl_modes = min(n_kl_modes,
                              len(self.eigenvalues_Kh),
                              len(self.eigenvalues_por))

        # Parameter counts:  20 + 20 + 2 + 2 + 1 = 45
        N = self.n_kl_modes
        self.no_parameters = 2 * N + 5
        self.no_observations = len(DEFAULT_OBSERVATION_TIMES_DAYS) * 4  # 72

        # Time stepping
        if time_steps is not None:
            self.time_steps = np.asarray(time_steps, dtype=float)
        else:
            # Default: whole year — 0.5 d × 34 + 2.0 d × 174 = 365 days
            self.time_steps = piecewise_uniform_time_steps([
                (0.5 * SEC_IN_DAY, 34),    # fine:   0–17 days
                (2.0 * SEC_IN_DAY, 174),   # coarse: 17–365 days
            ])

        # Observation extraction
        self.observation_times_days = DEFAULT_OBSERVATION_TIMES_DAYS.copy()
        self.pressure_unit_factor = PA_TO_MWH

        # Create reusable DG-0 function space
        self._Q_dg0 = dfx.fem.functionspace(self.mesh, ('DG', 0))

    def set_parameters(self, par):
        """Set material parameters from a flat vector of length 45.

        Layout (N = n_kl_modes = 20):
          par[ 0:N ]   xi_1 ... xi_N   KL coefficients for K_h GRF
          par[ N:2N]   xi_1 ... xi_N   KL coefficients for porosity GRF
          par[2N]      K_d inner  [Pa]
          par[2N+1]    K_d outer  [Pa]
          par[2N+2]    G inner    [Pa]
          par[2N+3]    G outer    [Pa]
          par[2N+4]    K_s        [Pa]
        """
        N = self.n_kl_modes
        self.xi_Kh  = par[:N]
        self.xi_por = par[N:2*N]
        self.K_d_inner = par[2*N]
        self.K_d_outer = par[2*N + 1]
        self.G_inner   = par[2*N + 2]
        self.G_outer   = par[2*N + 3]
        self.K_s       = par[2*N + 4]

    def get_observations(self):
        """Run the simulation and return observations.

        Returns
        -------
        obs : np.ndarray, shape (n_obs_times * 4,)
            Pressures at 4 sensors at each observation time, in metres
            of water head.
        """
        mesh = self.mesh
        Q = self._Q_dg0

        # --- Permeability field: GRF → K_h = 10^(GRF-13) → k = K_h/(ρ_f g) ---
        def _kh_transform(grf):
            return 10.0 ** (grf - 13) / (RHO_F * G_ACC)

        k_fnc = build_permeability_field(
            mesh, self.xi_Kh,
            self.eigenvalues_Kh, self.eigenvectors_Kh,
            transform=_kh_transform)

        # --- Porosity field: GRF → n = 10^(GRF-2.5) ---
        def _por_transform(grf):
            return 10.0 ** (grf - 2.5)

        n_fnc = build_permeability_field(
            mesh, self.xi_por,
            self.eigenvalues_por, self.eigenvectors_por,
            transform=_por_transform)

        # --- Zone-dependent K_d and G (sigmoid transition) ---
        K_d_fnc = build_sigmoid_field_dg0(mesh, self.K_d_inner, self.K_d_outer)
        G_fnc   = build_sigmoid_field_dg0(mesh, self.G_inner,   self.G_outer)

        # --- Derived quantities (all spatially varying, DG-0 arrays) ---
        K_d_arr = K_d_fnc.x.array
        G_arr   = G_fnc.x.array
        n_arr   = n_fnc.x.array
        K_s     = self.K_s

        alpha_arr = 1.0 - K_d_arr / K_s
        E_d_arr   = 9.0 * K_d_arr * G_arr / (3.0 * K_d_arr + G_arr)
        nu_d_arr  = (3.0 * K_d_arr - 2.0 * G_arr) / (2.0 * (3.0 * K_d_arr + G_arr))
        S_arr     = (alpha_arr - n_arr) / K_s + n_arr / K_F
        mu_arr    = G_arr
        lmbda_arr = E_d_arr * nu_d_arr / ((1.0 + nu_d_arr) * (1.0 - 2.0 * nu_d_arr))

        # Pack into DG-0 functions
        lmbda_fnc = dfx.fem.Function(Q);  lmbda_fnc.x.array[:] = lmbda_arr
        mu_fnc    = dfx.fem.Function(Q);  mu_fnc.x.array[:]    = mu_arr
        alpha_fnc = dfx.fem.Function(Q);  alpha_fnc.x.array[:] = alpha_arr
        cpp_fnc   = dfx.fem.Function(Q);  cpp_fnc.x.array[:]   = S_arr

        # --- Run simulation ---
        times, pressure_data = tsx_setup_and_computation_v2(
            mesh,
            lmbda_fnc, mu_fnc, alpha_fnc, cpp_fnc, k_fnc,
            time_steps=self.time_steps,
            sigma_xx=-SIGMA_X,
            sigma_yy=-SIGMA_Y,
        )

        # Reshape to (n_sensors, n_time_points)
        n_times = len(pressure_data)
        n_sensors = 4
        data_fp = np.zeros((n_sensors, n_times))
        for i, step_data in enumerate(pressure_data):
            data_fp[:, i] = [value[0] for value in step_data]

        # Interpolate to observation times
        times_days = times / SEC_IN_DAY
        obs_times = self.observation_times_days

        res = []
        for sensor_idx in range(n_sensors):
            p_interp = np.interp(obs_times, times_days, data_fp[sensor_idx, :])
            res.append(p_interp / self.pressure_unit_factor)

        # Store raw data for debugging / plotting
        self.data = pressure_data
        self.simulation_times = times

        return np.array(res).reshape((-1,))


    def plot_observations(self):
        """Plot the pressure time series at each sensor.
        Mark observation times with vertical dashed lines.
        """
        import matplotlib.pyplot as plt
        titles = ["HGT1-5", "HGT1-4", "HGT2-3", "HGT2-4"]
        # 4 subplots:
        _, axes = plt.subplots(2, 2, figsize=(10, 8))
        for i in range(4):
            ax = axes[i // 2, i % 2]
            # list indices must be integers or slices, not tuple:
            sensor_data = np.array([temp[i] for temp in self.data]) / self.pressure_unit_factor
            ax.plot(self.simulation_times / SEC_IN_DAY, sensor_data, label="simulated")
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Pressure (m water head)')
            ax.set_title(titles[i])
            for t in self.observation_times_days:
                ax.axvline(t, color='gray', linestyle='--', alpha=0.5)
            # add reference observations as red points
            obs_times = self.observation_times_days
            num_obs_times = len(obs_times)
            obs_values = observations[i*num_obs_times:(i+1)*num_obs_times]
            ax.scatter(obs_times, obs_values, color='red', label='reference')
            ax.legend()
        plt.suptitle('Simulated Pressure Time Series at Sensors')
        plt.tight_layout()