from pythtb import TBModel
import numpy as np

def kane_mele(
        onsite: int | float | np.integer | np.floating, 
        t: int | float | complex | np.integer | np.floating |  np.complexfloating, 
        soc: int | float | complex | np.integer | np.floating |  np.complexfloating, 
        rashba: int | float | complex | np.integer | np.floating |  np.complexfloating
        ) -> TBModel:
    """
    kane_mele tight-binding model.

    Parameters
    ----------
    onsite : int | float
        On-site energy.
    t : int | float | complex
        Hopping parameter.
    soc : int | float | complex
        Spin-orbit coupling strength.
    rashba : int | float | complex
        Rashba coupling strength.

    Returns
    -------
    TBModel
        An instance of the model.
    """

    # define lattice vectors
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    # define coordinates of orbitals
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

    # make two dimensional tight-binding Kane-Mele model
    ret_model = TBModel(2, 2, lat, orb, nspin=2)

    # set on-site energies
    ret_model.set_onsite([onsite, -onsite])

    # useful definitions
    sigma_x = np.array([0, 1, 0, 0])
    sigma_y = np.array([0, 0, 1, 0])
    sigma_z = np.array([0, 0, 0, 1])

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    
    # spin-independent first-neighbor hoppings
    ret_model.set_hop(t, 0, 1, [0, 0])
    ret_model.set_hop(t, 0, 1, [0, -1])
    ret_model.set_hop(t, 0, 1, [-1, 0])

    # second-neighbour spin-orbit hoppings (s_z)
    nnn_hop = 1j * soc * sigma_z
    ret_model.set_hop(-nnn_hop, 0, 0, [0, 1])
    ret_model.set_hop(nnn_hop,  0, 0, [1, 0])
    ret_model.set_hop(-nnn_hop, 0, 0, [1, -1])
    ret_model.set_hop(nnn_hop,  1, 1, [0, 1])
    ret_model.set_hop(-nnn_hop, 1, 1, [1, 0])
    ret_model.set_hop(nnn_hop,  1, 1, [1, -1])

    # Rashba first-neighbor hoppings: (s_x)(dy)-(s_y)(d_x)
 
    # bond unit vectors are (np.sqrt(3) / 2, 1/2) then (0,-1) then (-np.sqrt(3) / 2, 1/2)
    ret_model.set_hop(
        1j * rashba * ((1/2) * sigma_x - (np.sqrt(3) / 2) * sigma_y), 0, 1, [0, 0], mode="add"
    )
    ret_model.set_hop(1j * rashba * -sigma_x, 0, 1, [0, -1], mode="add")
    ret_model.set_hop(
        1j * rashba * ((1/2) * sigma_x + (np.sqrt(3) / 2) * sigma_y), 0, 1, [-1, 0], mode="add"
    )

    return ret_model
