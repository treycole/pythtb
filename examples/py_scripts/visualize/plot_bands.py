#!/usr/bin/env python

""" Toy graphene model
"""

from pythtb.models import graphene

# set model parameters
delta = 0.3
t = -1.0

my_model = graphene(delta=delta, t=t)

# generate list of k-points following a segmented path in the BZ
# list of nodes (high-symmetry points) that will be connected
path = [[0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [0.5, 0.5], [0.0, 0.0]]
# labels of the nodes
label = (r"$\Gamma $", r"$K$", r"$M$", r"$\Gamma $")
# total number of interpolated k-points along the path
nk = 200
# index of orbitals to project onto
proj_orb_idx = [0]

my_model.plot_bands(
    k_path=path, k_label=label, nk=nk, show=True, 
    proj_orb_idx=proj_orb_idx
    )