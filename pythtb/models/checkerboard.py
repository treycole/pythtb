from pythtb import TBModel

def checkerboard(t0, tprime, delta) -> TBModel:
    """
    checkerboard tight-binding model. 

    Parameters
    ----------
    t0 : float
        Nearest neighbor hopping amplitude.
    tprime : float
        Next-nearest neighbor hopping amplitude. Pierls phase is included.
        
    delta : float
        On-site energy. Positive for one sublattice, negative for the other.

    Returns
    -------
    TBModel
        An instance of the model.
    """

    lat = [[1.0, 0.0], [0.0, 1.0]]
    orb = [[0.0, 0.0], [0.5, 0.5]]

    model = TBModel(2, 2, lat=lat, orb=orb)

    # set on-site energies
    model.set_onsite([-delta, delta], mode='set')

    # set NN hoppings
    model.set_hop(-t0, 0, 0, [1, 0], mode='set')
    model.set_hop(-t0, 0, 0, [0, 1], mode='set')
    model.set_hop(t0, 1, 1, [1, 0], mode='set')
    model.set_hop(t0, 1, 1, [0, 1], mode='set')
    # set NNN hoppings
    model.set_hop(tprime, 1, 0, [1, 1], mode='set')
    model.set_hop(tprime*1j, 1, 0, [0, 1], mode='set')
    model.set_hop(-tprime, 1, 0, [0, 0], mode='set')
    model.set_hop(-tprime*1j, 1, 0, [1, 0], mode='set')

    return model