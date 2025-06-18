from pythtb import TBModel
from numpy import sqrt

def haldane(delta, t, t2):
    lat = [[1, 0], [0.5, sqrt(3)/2]]
    orb = [[1/3, 1/3], [2/3, 2/3]]

    model = TBModel(2, 2, lat, orb)

    model.set_onsite([-delta, delta], mode='set')

    for lvec in ([0, 0], [-1, 0], [0, -1]):
        model.set_hop(t, 0, 1, lvec, mode='set')

    for lvec in ([1, 0], [-1, 1], [0, -1]):
        model.set_hop(t2*1j, 0, 0, lvec, mode='set')
        model.set_hop(t2*-1j, 1, 1, lvec, mode='set')

    return model