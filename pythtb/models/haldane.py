from pythtb import TBModel
from numpy import sqrt


def haldane(delta: float, t: float, t2: float) -> TBModel:
    """
    haldane tight-binding model.

    Parameters
    ----------
    delta : float
        Onsite mass term. Opposite sign for the two sublattices.
    t : float
        Nearest neighbor hopping amplitude.
    t2 : float
        Next-nearest neighbor hopping amplitude. Peierls phase is included.

    Returns
    -------
    TBModel
        An instance of the model.
    """

    lat = [[1, 0], [0.5, sqrt(3) / 2]]
    orb = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]

    model = TBModel(2, 2, lat, orb)

    model.set_onsite([-delta, delta], mode="set")

    for lvec in ([0, 0], [-1, 0], [0, -1]):
        model.set_hop(t, 0, 1, lvec, mode="set")

    for lvec in ([1, 0], [-1, 1], [0, -1]):
        model.set_hop(t2 * 1j, 0, 0, lvec, mode="set")
        model.set_hop(t2 * -1j, 1, 1, lvec, mode="set")

    return model
