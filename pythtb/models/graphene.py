import numpy as np
from pythtb import TBModel

def graphene(delta:float, t:float) -> TBModel:
    """
    graphene tight-binding model.

    Parameters
    ----------
    delta : float
        On-site energy difference between the two orbitals.
    t : float
        Hopping parameter between nearest neighbor orbitals.

    Returns
    -------
    TBModel
        An instance of the model.
    """

        
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

    my_model = TBModel(2, 2, lat, orb)

    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])

    return my_model