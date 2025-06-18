import os
import numpy as np
import json
import datetime
import platform
import importlib.metadata
from run import run

OUTPUTDIR = "golden_outputs"
OUTPUTNAME = "result.npy" #NOTE: Replace with your expected output file name
LOGFILE = os.path.join(os.path.dirname(__file__), "golden_data_log.json")

def get_version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def regenerate():
    os.makedirs(OUTPUTDIR, exist_ok=True)

    #NOTE: Modify as needed to save expected output        
    result = run()
    np.save(os.path.join(OUTPUTDIR, OUTPUTNAME), result)

    metadata = {
        "generated_at": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "pythtb_version": get_version("pythtb")
    }
    with open(LOGFILE, "w") as f:
        json.dump(metadata, f, indent=4)

    print("✅ Golden data regenerated and logged to", LOGFILE)

if __name__ == "__main__":
    regenerate()
