from pythtb import TBModel
from numpy import sqrt
import numpy as np


def kane_mele(onsite, t, soc, rashba):
    """
    kane_mele tight-binding model.

    Parameters
    ----------
    onsite : TYPE
        Description.
    t : TYPE
        Description.
    soc : TYPE
        Description.
    rashba : TYPE
        Description.

    Returns
    -------
    TBModel
        An instance of the model.
    """

    # define lattice vectors
    lat = [[1.0, 0.0], [0.5, sqrt(3.0) / 2.0]]
    # define coordinates of orbitals
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

    # make two dimensional tight-binding Kane-Mele model
    ret_model = TBModel(2, 2, lat, orb, nspin=2)

    # set on-site energies
    ret_model.set_onsite([onsite, -onsite])

    # useful definitions
    sigma_x = np.array([0.0, 1.0, 0.0, 0])
    sigma_y = np.array([0.0, 0.0, 1.0, 0])
    sigma_z = np.array([0.0, 0.0, 0.0, 1])

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    # spin-independent first-neighbor hoppings
    ret_model.set_hop(t, 0, 1, [0, 0])
    ret_model.set_hop(t, 0, 1, [0, -1])
    ret_model.set_hop(t, 0, 1, [-1, 0])

    # second-neighbour spin-orbit hoppings (s_z)
    ret_model.set_hop(-1.0j * soc * sigma_z, 0, 0, [0, 1])
    ret_model.set_hop(1.0j * soc * sigma_z, 0, 0, [1, 0])
    ret_model.set_hop(-1.0j * soc * sigma_z, 0, 0, [1, -1])
    ret_model.set_hop(1.0j * soc * sigma_z, 1, 1, [0, 1])
    ret_model.set_hop(-1.0j * soc * sigma_z, 1, 1, [1, 0])
    ret_model.set_hop(1.0j * soc * sigma_z, 1, 1, [1, -1])

    # Rashba first-neighbor hoppings: (s_x)(dy)-(s_y)(d_x)
    r3h = np.sqrt(3.0) / 2.0
    # bond unit vectors are (r3h,half) then (0,-1) then (-r3h,half)
    ret_model.set_hop(
        1.0j * rashba * (0.5 * sigma_x - r3h * sigma_y), 0, 1, [0, 0], mode="add"
    )
    ret_model.set_hop(1.0j * rashba * (-1.0 * sigma_x), 0, 1, [0, -1], mode="add")
    ret_model.set_hop(
        1.0j * rashba * (0.5 * sigma_x + r3h * sigma_y), 0, 1, [-1, 0], mode="add"
    )

    return ret_model
