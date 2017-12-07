import sys
sys.path.append("../examples/")
sys.path.append("../")

from edge import *

def test_answer():
    print(evecs[ed,:])
    assert np.isclose(evecs[ed,0],2.80366220e-01+0.00000000e+00j)
    assert np.isclose(evecs[ed,4],1.56548374e-01 +3.48852597e-02j)
