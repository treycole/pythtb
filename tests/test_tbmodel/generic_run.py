import numpy as np
from pythtb import TBModel, WFArray


def generic_test_of_models(models: list[TBModel], use_dir, use_occ):

    # check that berry phases are the same
    val = []
    for ii, mod in enumerate(models):
        my_array = WFArray(mod, [11, 11])
        my_array.solve_on_grid([-0.5, -0.5])
        val.append(my_array.berry_phase(use_occ[ii], 1, contin=True))
    val = np.array(val)
    passed = []
    for i in range(1, val.shape[0]):
        passed.append(np.isclose(val[0], val[i]))
    passed = np.array(passed)
    assert np.all(passed)

    # check that energies are the same at some random point
    val = []
    for ii, mod in enumerate(models):
        val.append(mod.solve_ham([0.123, 0.523]))
    val = np.array(val)
    passed = []
    for i in range(1, val.shape[0]):
        passed.append(np.isclose(val[0], val[i]))
    passed = np.array(passed)
    assert np.all(passed)

    # check finitely cut models
    val = []
    H = []
    evecs = []
    for ii, mod in enumerate(models):
        mod_cut = mod.cut_piece(4, use_dir[ii], glue_edgs=False)
        H.append(mod_cut.get_ham([0.214]))
        evalu, evec = mod_cut.solve_ham([0.214], return_eigvecs=True)
        evecs.append(evec)
        val.append(mod_cut.position_expectation(evec, use_dir[ii]))
       

    val = np.array(val)
    passed = []
    for i in range(1, val.shape[0]):
        # only sum is multi-band gauge invariant (there are degeneracies)
        passed.append(np.isclose(np.sum(val[0]), np.sum(val[i])))
  
    passed = np.array(passed)
    assert np.all(passed)
