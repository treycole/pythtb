#!/usr/bin/env python

# Haldane model from Phys. Rev. Lett. 61, 2015 (1988)
# Calculates density of states for finite sample of Haldane model

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from pythtb.tb_model import TBModel  # import TB model class
import numpy as np
import matplotlib.pyplot as plt

# define lattice vectors
lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
# define coordinates of orbitals
orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

# make two dimensional tight-binding Haldane model
my_model = TBModel(2, 2, lat, orb)

# set model parameters
delta = 0.0
t = -1.0
t2 = 0.15 * np.exp((1.0j) * np.pi / 2.0)
t2c = t2.conjugate()

# set on-site energies
my_model.set_onsite([-delta, delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t, 0, 1, [0, 0])
my_model.set_hop(t, 1, 0, [1, 0])
my_model.set_hop(t, 1, 0, [0, 1])
# add second neighbour complex hoppings
my_model.set_hop(t2, 0, 0, [1, 0])
my_model.set_hop(t2, 1, 1, [1, -1])
my_model.set_hop(t2, 1, 1, [0, 1])
my_model.set_hop(t2c, 1, 1, [1, 0])
my_model.set_hop(t2c, 0, 0, [1, -1])
my_model.set_hop(t2c, 0, 0, [0, 1])

# print tight-binding model details
print(my_model)

# cutout finite model first along direction x with no PBC
tmp_model = my_model.cut_piece(20, 0, glue_edgs=False)
# cutout also along y direction
fin_model_false = tmp_model.cut_piece(20, 1, glue_edgs=False)

# cutout finite model first along direction x with PBC
tmp_model = my_model.cut_piece(20, 0, glue_edgs=True)
# cutout also along y direction
fin_model_true = tmp_model.cut_piece(20, 1, glue_edgs=True)

print("Starting diagonalization")
print(fin_model_false.nstate, fin_model_true.nstate)

# solve finite model
evals_false = fin_model_false.solve_ham()
evals_true = fin_model_true.solve_ham()

print("Finished diagonalization")

# flatten eigenvalue arrays
evals_false = evals_false.flatten()
evals_true = evals_true.flatten()

# now plot density of states
fig, ax = plt.subplots()
ax.hist(evals_false, 50, range=(-4.0, 4.0))
ax.set_ylim(0.0, 80.0)
ax.set_title("Finite Haldane model without PBC")
ax.set_xlabel("Band energy")
ax.set_ylabel("Number of states")
fig.tight_layout()
fig.savefig("haldane_fin_dos_false.pdf")
fig, ax = plt.subplots()
ax.hist(evals_true, 50, range=(-4.0, 4.0))
ax.set_ylim(0.0, 80.0)
ax.set_title("Finite Haldane model with PBC")
ax.set_xlabel("Band energy")
ax.set_ylabel("Number of states")
fig.tight_layout()
plt.show()

fig.savefig("haldane_fin_dos_true.pdf")

print("Done.\n")
