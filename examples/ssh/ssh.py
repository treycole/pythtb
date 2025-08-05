#!/usr/bin/env python

# one dimensional ssh chain
# demontrates new `TBModel.hamiltonian` function

# Copyright under GNU General Public License 2010, 2012, 2016
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from pythtb import TBModel
from pythtb.utils import pauli_decompose
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def ssh(v, w):
    lat = [[1]]
    orb = [[0], [1/2]]
    my_model = TBModel(1, 1, lat, orb)

    my_model.set_hop(v, 1, 0, [1])
    my_model.set_hop(w, 0, 1, [0])

    return my_model

v = -1         # intercell hopping
w_init = -.5  # initial intracell hopping

# define a path in k-space
(k_vec, k_dist, k_node) = ssh(v, w_init).k_path("full", 200)
k_label = [r"$0$", r"$\pi$", r"$2\pi$"]
print(k_vec)

fig, (ax1, ax2) = plt.subplots(1, 2)

model = ssh(v, w_init)
evals, evecs = model.solve_ham(k_vec, return_eigvecs=True)
ham = model.hamiltonian(k_vec)
print(ham[0], ham[-1])

# Compute phase difference
numerator = evecs[:, :, 1]
denominator = evecs[:, :, 0]
# Mask invalid (zero) denominators to avoid warnings
with np.errstate(divide='ignore', invalid='ignore'):
    phase_diff = np.angle(numerator / denominator)
    # Where both components are zero, set phase to NaN
    phase_diff[~np.isfinite(phase_diff)] = np.nan
    
scatters = []
for band in range(2):
    # Use scatter for per-point coloring
    scat = ax1.scatter(
        k_dist, evals[:, band], c=phase_diff[:, band], cmap='twilight_shifted',
        s=10, marker='o', label=f'Band {band}', vmin=-np.pi, vmax=np.pi)
    scatters.append(scat)

cbar = fig.colorbar(
    scat,
    ax=ax1,
    orientation='vertical',
    pad=0.01,
    ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
)
cbar.ax.set_yticklabels(
    [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
)
cbar.set_label("Phase difference  $\\arg(\\psi_1/\\psi_0)$", fontsize=11)

# plot path of endpoints of d-vec in dx dy plane as wave-vector sweeps from 0 to 2pi
d_vec = np.zeros((len(k_vec), 4), dtype=complex)
for k in range(len(k_vec)):
    d_vec[k] = pauli_decompose(ham[k])

# print(d_vec)
# Plot the path of the endpoints of d-vec in the dx-dy plane
dx = d_vec[:, 1].real
dy = d_vec[:, 2].real
# ax2.plot(np.append(dx, dx[0]), np.append(dy, dy[0]), 'b-')
ax2.plot(0, 0, 'ro')  # mark origin

ax2.plot(d_vec[:, 1].real, d_vec[:, 2].real, 'b-')

ax2.set_xlabel(r'$d_x$')
ax2.set_ylabel(r'$d_y$')
ax2.set_xlim(-2.5, 1.5)
ax2.set_ylim(-2.5, 2.5)
ax2.grid()

I = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

assert np.allclose(pauli_decompose(sigma_x), [0, 1, 0, 0])
assert np.allclose(pauli_decompose(sigma_y), [0, 0, 1, 0])
assert np.allclose(pauli_decompose(sigma_z), [0, 0, 0, 1])

# check that the d_vec decomp reproduces the hamiltonian
for k in range(len(k_vec)):
    assert np.allclose(ham[k], d_vec[k, 0] * I + d_vec[k, 1] * sigma_x + d_vec[k, 2] * sigma_y + d_vec[k, 3] * sigma_z)

# wf_arr = WFArray(model, [100])
# wf_arr.solve_on_grid(k_vec)
# berry_phase = wf_arr.berry_phase(occ=[0], berry_evals=False)
# wfc = berry_phase[0] / (2 * np.pi)
# ax2.plot([w_init], [wfc], 'ro')
# ax2.set_xlabel(r'intracell hopping $w$')
# ax2.set_ylabel(r'Wannier center $\bar{x}$')
# ax2.set_title(r"Wannier center vs $w$")

ax1.set_title("1D SSH chain band structure" + "\n" + fr"$v = {v:.2f}$, $w = {w_init:.2f}$")
ax1.set_xlabel("Path in k-space")
ax1.set_ylabel("Band energy")
ax1.set_xticks(k_node)
ax1.set_xticklabels(k_label)
ax1.set_xlim(k_node[0], k_node[-1])
for n in range(len(k_node)):
    ax1.axvline(x=k_node[n], linewidth=0.5, color="k")
fig.tight_layout(rect=[0, 0.07, 1, 1])

# Make a slider for w (intracell hopping)
ax_slider = plt.axes([0.15, 0.01, 0.7, 0.03])  # [left, bottom, width, height]
w_slider = Slider(
    ax=ax_slider,
    label=r"intracell hopping $w$",
    valmin=-2.0,
    valmax=2.0,
    valinit=w_init,
    valstep=0.01,
)

w_values = []
wfc_values = []

def update(val):
    w = w_slider.val
    model = ssh(v, w)
    evals, evecs = model.solve_ham(k_vec, return_eigvecs=True)
    ham = model.hamiltonian(k_vec)

    numerator = evecs[:, :, 1]
    denominator = evecs[:, :, 0]
    # Mask invalid (zero) denominators to avoid warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        phase_diff = np.angle(numerator / denominator)
        # Where both components are zero, set phase to NaN
        phase_diff[~np.isfinite(phase_diff)] = np.nan

    # Remove previous scatters
    for scat in scatters:
        scat.remove()
    scatters.clear()

    # Re-plot scatters with updated colors
    for band in range(2):
        scat = ax1.scatter(
            k_dist, evals[:, band], c=phase_diff[:, band], cmap='twilight_shifted',
            s=10, marker='o', label=f'Band {band}', vmin=-np.pi, vmax=np.pi)
        scatters.append(scat)
    
    ax1.set_title("1D SSH chain band structure" + "\n" + fr"$v = {v:.2f}$, $w = {w:.2f}$")

    # update d-vec
    d_vec = np.zeros((len(k_vec), 4), dtype=complex)
    for k in range(len(k_vec)):
        d_vec[k] = pauli_decompose(ham[k])

    # Plot the path of the endpoints of d-vec in the dx-dy plane
    ax2.cla()
    dx = d_vec[:, 1].real
    dy = d_vec[:, 2].real
    # ax2.plot(np.append(dx, dx[0]), np.append(dy, dy[0]), 'b-')
    ax2.plot(0, 0, 'ro')  # mark origin
    ax2.plot(d_vec[:, 1].real, d_vec[:, 2].real, 'b-')
    ax2.set_xlabel(r'$d_x$')
    ax2.set_ylabel(r'$d_y$')
    ax2.set_title("Path of $d$-vector in $d_x$-$d_y$ plane")
    ax2.set_xlim(-2.5, 1.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid()

    # # Recompute Wannier center
    # wf_arr = WFArray(model, [100])
    # wf_arr.solve_on_grid(k_vec)
    # berry_phase = wf_arr.berry_phase(occ=[0], berry_evals=False)
    # wfc = berry_phase[0] / (2 * np.pi)

    # w_values.append(w)
    # wfc_values.append(wfc)
    
    # # Update ax2
    # ax2.cla()  # clear previous
    # ax2.plot(w_values, wfc_values, 'ro')
    # ax2.set_xlabel(r'intracell hopping $w$')
    # ax2.set_ylabel(r'Wannier center $\bar{x}$')
    # ax2.set_title(r"Wannier center vs $w$")

    fig.canvas.draw_idle()

w_slider.on_changed(update)

plt.show()