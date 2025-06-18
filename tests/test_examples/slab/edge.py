#!/usr/bin/env python

# Haldane model from Phys. Rev. Lett. 61, 2015 (1988)
# Solves model and draws one of its edge states.

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from pythtb import TBModel 
import numpy as np

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

# cutout finite model first along direction x with no PBC
tmp_model = my_model.cut_piece(10, 0, glue_edgs=False)
# cutout also along y direction with no PBC
fin_model = tmp_model.cut_piece(10, 1, glue_edgs=False)

# cutout finite model first along direction x with PBC
tmp_model_half = my_model.cut_piece(10, 0, glue_edgs=True)
# cutout also along y direction with no PBC
fin_model_half = tmp_model_half.cut_piece(10, 1, glue_edgs=False)

# solve finite models
(evals, evecs) = fin_model.solve_ham(return_eigvecs=True)
(evals_half, evecs_half) = fin_model_half.solve_ham(return_eigvecs=True)

# pick index of state in the middle of the gap
ed = fin_model.get_num_orbitals() // 2