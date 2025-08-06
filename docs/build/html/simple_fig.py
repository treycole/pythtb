from pythtb import TBModel
import numpy as np  
import matplotlib.pyplot as plt

# lattice vectors and orbital positions
lat = [[1, 0], [1/2, np.sqrt(3)/2]]
orb = [[1/3, 1/3], [2/3, 2/3]]

# two-dimensional tight-binding model
model = TBModel(2, 2, lat, orb)

# define hopping between orbitals
model.set_hop(-1.0, 0, 1, [ 0, 0])
model.set_hop(-1.0, 1, 0, [ 1, 0])
model.set_hop(-1.0, 1, 0, [ 0, 1])

# solve model on a path in k-space
k_nodes = [[0.0, 0.0],[1./3., 2./3.],[0.5,0.5]]
nk = 100  # number of k-points
# k_vec: k-points, k_dist: distance along path, k_node: node
(k_vec, k_dist, k_node) = model.k_path(k_nodes, nk, report=False)
evals = model.solve_ham(k_vec)

# plot bandstructure
fig, ax = plt.subplots()
ax.plot(k_dist, evals)
ax.set_xticks(k_node)
ax.set_xticklabels([r"$\Gamma$", r"$K$", r"$M$"])
ax.set_xlim(k_node[0], k_node[-1])
fig.savefig("band.png")
