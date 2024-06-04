# Code used to calculate DAP Fittings
# Refracted on January 16, 2024
# Change Headers of all the files in the project with contact info for IP, check what other folks are doing


from sample_lattices import *

E = 1.60217663 * 10 ** (-19)  # Units Coulombs
Eps_0 = 8.8541878128 * 10 ** (-12)  # Units C/V*m


# Generates Fit using fit glass, coincidence calc class, as well as
# raw data (E_s in text) and fit energies (E_DA n)
class Generate_Fit:
    def __init__(self, fit_class, coincidence_calc, raw_data, fit_energies):
        self.fit_class = fit_class
        self.coincidence_calc = coincidence_calc
        self.raw_data = raw_data
        self.fit_energies = fit_energies

    def run(self):
        det_idx_matrix = []
        det_val_matrix = []
        if len(np.asarray(self.raw_data)) > 0:
            raw_data = list(self.raw_data)
            for e in self.fit_energies:
                test_data = self.fit_class.set(e)
                det_idx, det_val = self.coincidence_calc.val(test_data, raw_data)
                det_idx_matrix.append(np.asarray(det_idx))
                det_val_matrix.append(np.asarray(det_val))
        M_mat = reshape_detuning(det_val_matrix, det_idx_matrix, self.fit_class)
        return M_mat


class Co_Match:
    def __init__(self, bounds, blocking):
        self.bounds = bounds
        self.blocking = blocking

    def val(self, energy, data):
        data = sorted(data, reverse=True)
        energy_c = energy.copy()
        test_c = data.copy()
        e_idx_full = [j for j, el in enumerate(energy_c)]
        e_idx_full = [e_idx_full[j] for j, el in enumerate(energy_c) if self.bounds[0] < el < self.bounds[1]]
        energy_c = [el for el in energy_c if self.bounds[0] < el < self.bounds[1]]
        e_idx_calc = e_idx_full.copy()
        mu_arr = []
        delta_arr = []
        a, b = np.meshgrid(energy_c, test_c)
        D_mat = np.abs(a - b)
        theta_r = max(np.abs(data[0] - energy[-1]), np.abs(data[-1] - energy[0]))
        if D_mat.shape[0] > 1 and D_mat.shape[1] > 0:
            while np.min(D_mat) < theta_r:
                m_index = np.unravel_index(D_mat.argmin(), D_mat.shape)
                delta_arr.append(D_mat[m_index])
                mu_arr.append(e_idx_calc[m_index[1]])
                D_mat = self.blocking(m_index, D_mat, theta_r)
        return mu_arr, delta_arr


def energy_from_distance(dist, en, dielectric):
    # Multiplied by 10^13 because of lattice maker code returning distances in units of 10^-13m
    k = E * 10 ** 13 / (4 * np.pi * Eps_0 * dielectric)  # Only one unit of e to keep in eV
    return en + np.divide(k, dist)

# hBN Dielectric = 6.93
class DAP_Set:
    def __init__(self, dist_array, dielectric):
        self.dielectric = dielectric
        self.dap_set = energy_from_distance(dist_array, 0, dielectric=dielectric)

    def set(self, e):
        test_data = self.dap_set + e
        return sorted(test_data, reverse=True)

class DAP_Set_3D:
    def __init__(self, dap_set):
        self.dap_set = dap_set
    def set(self, e):
        test_data = self.dap_set + e
        return sorted(test_data, reverse=True)


# Blocks off Individual Elements which have been matched from being matched again
def plus_blocking(m_index, mat, theta_r):
    mat[m_index[0], :] = theta_r
    mat[:, m_index[1]] = theta_r
    return mat


# Prevents Cross-Matching
def checker_blocking(m_index, mat, theta_r):
    mat[:m_index[0], m_index[1]:] = theta_r
    mat[m_index[0]:, :m_index[1]] = theta_r
    mat[m_index[0], :] = theta_r
    mat[:, m_index[1]] = theta_r
    return mat


# Reshaping data into NaN matrix for faster calculation in post
def reshape_detuning(det, det_idx, fit_class):
    res_array = np.full([len(det), len(fit_class.dap_set)], np.nan)
    for i, row in enumerate(det_idx):
        for j, col in enumerate(row):
            res_array[i, col] = det[i][j]
    return res_array


# Detuning Calculation from Fitting Matrix
# TODO: Rename variables here and then rename variables in supplementary
def det_calc(det_mat, w_co):
    det_mat_c = det_mat.copy()
    det_mat_c[np.isnan(det_mat_c)] = 0
    det_mat_c[np.where((det_mat_c > w_co))] = 0
    det_sum_arr = np.sum(det_mat_c, axis=1)
    norm_array = co_calc(det_mat, w_co) + 1  # Don't Divide By Zero
    det_sum_arr = np.divide(det_sum_arr, norm_array)
    return det_sum_arr


# Coincidence Calculation on Detuning Matrix
def co_calc(det_mat, w_co, penalty_func_class=None):
    det_mat2 = det_mat.copy()
    det_mat2[np.where((det_mat2 > w_co))] = np.NaN
    det_mat2[np.isnan(det_mat2)] = np.inf
    det_mat2[np.where((det_mat2 < w_co))] = 1
    det_mat2[np.where((det_mat2 != 1))] = 0
    det_mat2 = Co_Penalty(penalty_func_class).apply(det_mat2)
    co_arr = np.sum(det_mat2, axis=1)
    return co_arr


# Example ArcTan Penalty Function for Coincidence Calculation
class ArcTan_Penalty:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def apply(self, m):
        return (1 / 2) * (1 - (2 / np.pi) * np.arctan(self.a * (m - self.b)))


# Example Step Penalty Function for Coincidence Calculation
class Step_Penalty:
    def __init__(self, cutoff):
        self.cutoff = cutoff

    def apply(self, m):
        arr = np.ones(len(m))
        idx = [j for j, el in enumerate(m) if el >= self.cutoff]
        arr[idx] = 0
        return arr


# Class that handles applying penalty function [p(m)] methods to coincidence
class Co_Penalty:
    def __init__(self, penalty_func_class=None):
        self.penalty_func_class = penalty_func_class

    def apply(self, mat):
        if self.penalty_func_class is not None:
            cols = np.asarray(range(mat.shape[1]))
            cols = self.penalty_func_class.apply(cols)
            res_mat = np.zeros((mat.shape[0], mat.shape[1]))
            for row in range(res_mat.shape[0]):
                res_mat[row, :] = cols
            result = np.multiply(mat, res_mat)
        else:
            result = mat
        return result


def main():
    file = r"C:\Users\enriq\Downloads\GaN_mp-804_computed2.xyz"
    dists = general_lattice_calculator_xyz(file)[0:100]
    dists = dists[0:100]
    dists2 = dists[0:50]
    sim_data = energy_from_distance(dists2, 1.5, dielectric=8.9)
    fit_energies = np.linspace(1, 2, 10000)
    fit_class = DAP_Set(dists, dielectric=8.9)
    coincidence_calc = Co_Match([0, np.inf], checker_blocking)
    M_mat = Generate_Fit(fit_class, coincidence_calc, sim_data, fit_energies).run()
    co = co_calc(M_mat, .001, Step_Penalty(50))
    det = det_calc(M_mat, np.inf)
    plt.scatter(fit_energies, co)
    plt.show()

    plt.scatter(fit_energies, det)
    plt.show()


def main2():
    dists = hex_lattice_maker(11, a=2.51, c=6.66)[0:100]
    sim_data = energy_from_distance(dists, 1.5, dielectric=8.9)
    fit_energies = np.linspace(1, 2, 10000)
    fit_class = DAP_Set(dists, dielectric=8.9)
    coincidence_calc = Co_Match([0, np.inf], checker_blocking)
    M_mat = Generate_Fit(fit_class, coincidence_calc, sim_data, fit_energies).run()
    print(M_mat[6383, :])
    co = co_calc(M_mat, .001, Step_Penalty(50))
    print(M_mat[6383, :])
    det = det_calc(M_mat, np.inf)
    print(M_mat[6383, :])



if __name__ == '__main__':

    main2()
