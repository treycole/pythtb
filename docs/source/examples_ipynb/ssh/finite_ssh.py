#!/usr/bin/env python

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from pythtb import TBModel
import matplotlib.pyplot as plt
import numpy as np

def ssh(v, w):
    lat = [[1]]
    orb = [[0], [1/2]]
    my_model = TBModel(1, 1, lat, orb)

    my_model.set_hop(v, 0, 1, [0])
    my_model.set_hop(w, 1, 0, [1])

    return my_model

w_values = np.linspace(0, 1.0, 100)
v = 0.5
evals_w = []
evecs_w = []
n_cells = 20
for w in w_values:
    finite_model = ssh(v, w).cut_piece(n_cells, 0)
    evals, evecs = finite_model.solve_ham(return_eigvecs=True)
    # ham = finite_model.hamiltonian(k_vec)
    evals_w.append(evals)
    evecs_w.append(evecs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the eigenvalues
ax1.plot(w_values, evals_w, c='r')

# Mark the v hopping on the plot
ax1.axvline(x=v, color='b', linestyle='--', lw=1, label=r'$v$ hopping')

ax1.set_xlabel(r"$w$")
ax1.set_ylabel(r"$E$")
ax1.set_xlim(0, 1.0)
ax1.set_ylim(-2, 2)
ax1.set_title(f"Finite SSH Chain with {n_cells} unit cells")
ax1.legend()

# Plot edge state density at last w=1.0
band_idx = 20
density = np.abs(evecs_w[-1][band_idx, :])**2
position = np.arange(len(density)) / 2
ax2.plot(position, density)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(rf"$\psi_{{{band_idx}}}(x)$")
ax2.set_title(r"Edge state density at $w=1.0$")

plt.show()

