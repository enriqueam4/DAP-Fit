# January 17, 2024 Enrique Mejia
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


# Returns Hexagonal Lattice Distances in ordered list in units of 10^(-13) meters
# Input Lattice Constants in Angstroms
# Default values of a and c are set to those of hexagonal boron nitride [REF]
def hex_lattice_maker(lat_ext, a=2.51, c=6.66):
    vec_a1 = a * np.array([(np.sqrt(3)) / 2, 1 / 2, 0])
    vec_a2 = a * np.array([(np.sqrt(3)) / 2, -1 / 2, 0])
    vec_a3 = c * np.array([0, 0, 1 / 2])
    vec_B2N = (a / np.sqrt(3)) * np.array([-1, 0, 0])
    r_vec = []
    for layer in [0, 1]:
        for x in range(-lat_ext, lat_ext):
            for y in range(-lat_ext, lat_ext):
                for element in [1, 2]:
                    dist = x * vec_a1 + y * vec_a2 + vec_B2N * (element - 1) + vec_a3 * layer
                    Ar_m = np.sqrt(sum(dist * dist))
                    # Round to 3 decimal places and then convert to integer in units of 1/1000th of an angstrom
                    r_vec.append(int(round(Ar_m, 3) * 1000))
    r_vec = sorted(r_vec)
    r_vec = set(r_vec)
    r_vec.discard(0)
    r_vec = sorted(r_vec)
    return r_vec


# Returns Diamond Lattice Distances in ordered list in units of 10^(-13) meters
# Takes in lattice constant in units of Angstroms
# Default Lattice Constant Set to that of Silicon
def diamond_lattice_maker(lat_ext, a=5.43):
    a1 = (a / 2) * np.array([0, 1, 1])
    a2 = (a / 2) * np.array([1, 0, 1])
    a3 = (a / 2) * np.array([1, 1, 0])
    vec_offset = (a / 4) * np.array([1, 1, 1])
    r_vec = []
    for x in range(-lat_ext, lat_ext):
        for y in range(-lat_ext, lat_ext):
            for z in range(-lat_ext, lat_ext):
                for lat in [0, 1]:
                    dist = x * a1 + y * a2 + + z * a3 + vec_offset * (lat)
                    Ar_m = np.sqrt(sum(dist * dist))
                    # Round to 3 decimal places and then convert to integer in units of 1/1000th of an angstrom
                    r_vec.append(int(round(Ar_m, 3) * 1000))
    r_vec = sorted(r_vec)
    r_vec = set(r_vec)
    r_vec.discard(0)
    r_vec = sorted(r_vec)
    return r_vec


# Returns Hexagonal Lattice Distances in ordered list in units of 10^(-13) meters
# Takes in lattice constant in units of Angstroms
# Default Lattice Constant Set to that of Gallium Nitride
def wurtzite_lattice_maker(lat_ext, a=3.21629, c=5.23996, u=3/8):
    a1 = (a / 2) * np.asarray([1, -np.sqrt(3), 0])
    a2 = (a / 2) * np.asarray([1, +np.sqrt(3), 0])
    a3 = c * np.asarray([0, 0, 1])
    vec_offset = (2 / 3) * a1 + (1 / 3) * a2 + (1 / 2) * a3
    vec_offset2 = u * a3
    r_vec = []
    for x in range(-lat_ext, lat_ext):
        for y in range(-lat_ext, lat_ext):
            for z in range(-lat_ext, lat_ext):
                for lat in [0, 1]:
                    for lat2 in [0, 1]:
                        dist = x * a1 + y * a2 + + z * a3 + vec_offset * lat + vec_offset2 * lat2
                        Ar_m = np.sqrt(sum(dist * dist))
                        # Round to 3 decimal places and then convert to integer in units of 1/1000th of an angstrom
                        r_vec.append(int(round(Ar_m, 3) * 1000))
    r_vec = sorted(r_vec)
    r_vec = set(r_vec)
    r_vec.discard(0)
    r_vec = sorted(r_vec)
    return r_vec


def general_lattice_calculator_xyz(file):
    atomlist = generate_atom_list_from_xyz(file)
    r_vec = []
    for i, el1 in enumerate(atomlist):
        for j, el2 in enumerate(atomlist):
            r_vec.append(el1.find_distance(el2))
    r_vec = sorted(r_vec)
    r_vec = set(r_vec)
    r_vec.discard(0)
    r_vec = sorted(r_vec)
    return r_vec


def generate_atom_list_from_xyz(file):
    file1 = open(file, 'r')
    Lines = file1.readlines()
    data = []
    for line in Lines:
        data.append(line.strip('\n'))
    data.pop(0)
    data.pop(0)
    atomlist = []
    for el in data:
        tmp = el.split('  ')
        for i, el in enumerate(tmp):
            el.strip(' ')
            if el == "":
                tmp.pop(i)
        a = atom(float(tmp[1].strip(' ')), float(tmp[2].strip(' ')), float(tmp[3].strip(' ')), tmp[0].strip(' '))
        atomlist.append(a)
    return atomlist


class atom:
    def __init__(self, x, y, z, species):
        self.x = x
        self.y = y
        self.z = z
        self.species = species

    def find_distance(self, atom2):
        return np.sqrt((self.x - atom2.x) ** 2 + (self.y - atom2.y) ** 2 + (self.z - atom2.z) ** 2).round(4) * 1000


# Creates energies in eV from series of distances as well as transition energy
# Great for creating synthetic data for testing or for fitting
def energy_from_distance(dist, en, dielectric=6.93):
    e = 1.60217663 * 10 ** (-19)  # Units C
    pi = 3.13159265353  # Unitless
    eps_0 = 8.8541878128 * 10 ** (-12)  # Units C/V*m
    k = e * 10 ** 13 / (4 * pi * eps_0 * dielectric)  # Only one unit of e to keep in eV
    # Times 10^13 because of saved value of distance in 10^-13m (10^-3 A)
    # energy = en + k / dist
    energy = np.zeros(len(dist))
    for i in range(len(dist)):
        energy[i] = en + k / dist[i]
    return energy


if __name__ == '__main__':
    dists = wurtzite_lattice_maker(11)[:100]
    ens = energy_from_distance(dists, 0)
