import os
import numpy as np
from .run import run

OUTPUTDIR = "golden_outputs"

def regenerate():
    os.makedirs(OUTPUTDIR, exist_ok=True)
    result = run()
    np.save(os.path.join(OUTPUTDIR, "result.npy"), result)
    print("Golden data regenerated in", OUTPUTDIR)

if __name__ == "__main__":
    regenerate()
