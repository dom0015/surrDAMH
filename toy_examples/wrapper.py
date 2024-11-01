
from tunnel_with_subdomains import tsx_setup_and_computation, prepare_coefficient_functions, load_mesh_and_domain_tags

import matplotlib.pyplot as plt
import numpy as np


class SolverTSX():
    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = 9*4+2
        self.no_observations = 18*4

        self.mesh, self.cell_tags, _ = load_mesh_and_domain_tags("tsx_ellipses_regions_coarse")
        # hardcoded number of domains, 'constant' depending on the used mesh
        self.number_of_subdomains = 9  # TODO: could be taken from the mesh
        self.alpha_values = [0.2] * self.number_of_subdomains

    def set_parameters(self, par):
        NS = self.number_of_subdomains
        self.permeability_values = par[:NS]
        self.storativity_values = par[NS:2*NS]
        young_values = par[2*NS:3*NS]
        poisson_values = par[3*NS:4*NS]
        self.sigma_x = par[4*NS]
        self.sigma_y = par[4*NS+1]

        self.mu_values = young_values / (2 * (1 + poisson_values))
        self.lmbda_values = young_values * poisson_values / ((1 + poisson_values) * (1 - 2 * poisson_values))

        # young_e = 6e10
        # poisson_nu = 0.2
        # mu = young_e / (2 * (1 + poisson_nu))
        # lmbda = young_e * poisson_nu / ((1 + poisson_nu) * (1 - 2 * poisson_nu))
        # cpp = 7.712e-12

        # self.lmbda_values = [lmbda + _ for _ in range(number_of_subdomains)]
        # self.mu_values = [mu + _ for _ in range(number_of_subdomains)]
        # self.alpha_values = [alpha - 0.00001*_ for _ in range(number_of_subdomains)]
        # self.cpp_values = [cpp + _*1.0e-14 for _ in range(number_of_subdomains)]
        # self.k_values = [6.0e-19 + _*1.0e-20 for _ in range(number_of_subdomains)]

    def get_observations(self):
        lambda_fnc, mu_fnc, alpha_fnc, cpp_fnc, k_fnc = prepare_coefficient_functions(self.mesh, self.cell_tags,
                                                                                      self.lmbda_values, self.mu_values, self.alpha_values,
                                                                                      self.storativity_values, self.permeability_values)

        data = tsx_setup_and_computation(self.mesh,
                                         lambda_fnc, mu_fnc, alpha_fnc, cpp_fnc, k_fnc, sigma_xx=-self.sigma_x, sigma_yy=-self.sigma_y,
                                         tau_f=24*60*60/2, t_steps_num=358*2)
        data_fp = np.zeros((4, len(data)))
        for i, item in enumerate(data):
            data_fp[:, i] = [value[0] for value in data[i]]

        # names = ['HGT1-5', 'HGT1-4', 'HGT2-3', 'HGT2-4']
        res = []
        for i, timeline in enumerate(data_fp):
            # print(timeline)
            res.append(timeline[17*2::40]/9806)

        self.data = data
        return np.array(res).reshape((-1,))


observations = [754.64805252, 755.01945387, 655.95978987, 594.39699416,
                530.43745741, 515.8816575, 494.97253552, 475.59152216,
                452.86384229, 433.26618307, 382.97842896, 351.99284788,
                332.17098259, 309.92537921, 300.5676362, 303.92681608,
                287.90724713, 291.92276435, 515.43370038, 587.81872627,
                600.23143393, 604.48910264, 600.39683158, 599.11176231,
                603.23124273, 599.23918313, 599.03415368, 601.54081621,
                584.89274096, 571.77321889, 559.71106412, 551.93560232,
                550.11891225, 547.36679965, 546.21690982, 542.02776456,
                182.31709971, 199.55739055, 212.15676692, 224.15477335,
                229.70971358, 238.39284514, 253.75478535, 262.23129018,
                268.99356638, 276.3197908, 278.0979942, 281.01668497,
                280.27204155, 284.22676206, 286.21291704, 290.6920895,
                294.08053105, 294.73164677,  48.08262448,  42.93917678,
                49.94181195,  61.35265637,  59.6642977,  75.47073607,
                81.53250879,  90.51661055,  90.82438687,  89.70334219,
                79.27229886,  79.20607144,  83.22591843,  77.02257697,
                78.10138551,  84.91265489,  75.86837226,  82.69544488]

observations = np.array(observations)
