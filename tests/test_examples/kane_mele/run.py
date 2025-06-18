from model import get_kane_mele 
import numpy as np
from pythtb import TBModel, WFArray

def run_kane_mele():

    evals_list = []
    wan_cent_list = []
    for top_index in ["even", "odd"]:

        my_model = get_kane_mele(top_index)
        my_array = WFArray(my_model, [41, 41])
        my_array.solve_on_grid([-0.5, -0.5])
        
        # [Gamma, K, M, K', Gamma] path in the BZ
        path = [
            [0, 0],
            [2/3, 1/3],
            [1/2, 1/2],
            [1/3, 2/3],
            [0, 0],
        ]
        k_vec, _, _ = my_model.k_path(path, 101, report=False)

        evals = my_model.solve_ham(k_vec)
        evals_list.append(evals)

        wan_cent = my_array.berry_phase([0, 1], dir=1, contin=False, berry_evals=True)
        wan_cent /= 2.0 * np.pi
        wan_cent_list.append(wan_cent)

    return np.array(evals_list), np.array(wan_cent_list)
