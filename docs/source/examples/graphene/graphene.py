#!/usr/bin/env python

# Toy graphene model

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from pythtb import TBModel 
import numpy as np
import matplotlib.pyplot as plt

# define lattice vectors
lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
# define coordinates of orbitals
orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

# make two dimensional tight-binding graphene model
my_model = TBModel(2, 2, lat, orb)

# set model parameters
delta = 0.0
t = -1.0

# set on-site energies
my_model.set_onsite([-delta, delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t, 0, 1, [0, 0])
my_model.set_hop(t, 1, 0, [1, 0])
my_model.set_hop(t, 1, 0, [0, 1])

# print tight-binding model
my_model.report()

# generate list of k-points following a segmented path in the BZ
# list of nodes (high-symmetry points) that will be connected
path = [[0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [0.5, 0.5], [0.0, 0.0]]
# labels of the nodes
label = (r"$\Gamma $", r"$K$", r"$M$", r"$\Gamma $")
# total number of interpolated k-points along the path
nk = 121

# call function k_path to construct the actual path
k_vec, k_dist, k_node = my_model.k_path(path, nk)
# outputs:
#   k_vec: list of interpolated k-points
#   k_dist: horizontal axis position of each k-point in the list
#   k_node: horizontal axis position of each original node

print("---------------------------------------")
print("starting calculation")
print("---------------------------------------")
print("Calculating bands...")

# obtain eigenvalues to be plotted
evals = my_model.solve_ham(k_vec)

# figure for bandstructure
fig, ax = plt.subplots()

ax.set_xlim(k_node[0], k_node[-1])
ax.set_xticks(k_node)
ax.set_xticklabels(label)

for n in range(len(k_node)):
    ax.axvline(x=k_node[n], linewidth=0.5, color="k")
    
ax.set_title("Graphene band structure")
ax.set_xlabel("Path in k-space")
ax.set_ylabel("Band energy")

# plot bands
ax.plot(k_dist, evals)

# make an PDF figure of a plot
fig.tight_layout()
fig.savefig("graphene.pdf")
plt.show()

print("Done.\n")
