from pythtb import TBModel

def checkerboard(t0, tprime, delta):
    # define lattice vectors
    lat=[[1.0, 0.0], [0.0, 1.0]]
    # define coordinates of orbitals
    orb=[[0.0, 0.0], [0.5, 0.5]]

    # make two dimensional tight-binding checkerboard model
    model = TBModel(2, 2, lat=lat, orb=orb)

    # set on-site energies
    model.set_onsite([-delta, delta], mode='set')

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    model.set_hop(-t0, 0, 0, [1, 0], mode='set')
    model.set_hop(-t0, 0, 0, [0, 1], mode='set')
    model.set_hop(t0, 1, 1, [1, 0], mode='set')
    model.set_hop(t0, 1, 1, [0, 1], mode='set')

    model.set_hop(tprime, 1, 0, [1, 1], mode='set')
    model.set_hop(tprime*1j, 1, 0, [0, 1], mode='set')
    model.set_hop(-tprime, 1, 0, [0, 0], mode='set')
    model.set_hop(-tprime*1j, 1, 0, [1, 0], mode='set')

    return model