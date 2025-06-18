import sys

sys.path.append("../examples/")
sys.path.append("../")

from tests.test_examples.graphene.cone import w_square
import numpy as np


def test_answer():
    assert np.isclose(w_square.berry_flux([0]), 2.17921648013)
    assert np.isclose(w_square.berry_flux([1]), -2.17921648013)
    assert np.isclose(w_square.berry_flux([0, 1]), 0.0)
