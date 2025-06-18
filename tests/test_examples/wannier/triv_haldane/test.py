import os
import numpy as np
import datetime
import json
import platform
from .run import run

OUTPUTDIR = "golden_outputs"

def test_example():
    name = os.path.basename(os.path.dirname(__file__))
    group = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
    base_path = os.path.join("tests", "test_examples", group)
    log_file = os.path.join(base_path, "status.json")

    result = run()
    golden_path = os.path.join(os.path.dirname(__file__), OUTPUTDIR, "result.npy")
    expected = np.load(golden_path)

    entry = {
        "last_pass": datetime.datetime.now().isoformat(),
        "pythtb_version": get_version("pythtb"),
        "python_version": platform.python_version(),
        "status": "pass"
    }

    try:
        assert np.allclose(result, expected, rtol=1e-8)
    except AssertionError as e:
        entry["status"] = "fail"
        entry["reason"] = str(e)

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            status = json.load(f)
    else:
        status = {}

    status[name] = entry
    with open(log_file, "w") as f:
        json.dump(status, f, indent=4)

    if entry["status"] == "fail":
        raise AssertionError(entry["reason"])

def get_version(pkg):
    try:
        import importlib.metadata as im
        return im.version(pkg)
    except:
        return "unknown"
