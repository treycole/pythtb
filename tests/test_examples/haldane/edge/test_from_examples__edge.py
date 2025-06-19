import numpy as np
from tests.utils import import_run
import os

example_dir = os.path.dirname(__file__)
run = import_run(example_dir)

# from tests.test_examples.slab.edge.edge import evecs, ed
# def test_answer():
#     print(evecs[ed, :])
#     assert np.isclose(evecs[ed, 0], 2.80366220e-01 + 0.00000000e00j)
#     assert np.isclose(evecs[ed, 4], 1.56548374e-01 + 3.48852597e-02j)
