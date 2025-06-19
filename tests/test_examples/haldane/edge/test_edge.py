import os
import numpy as np
import json
import datetime
import platform
from tests.utils import import_run

OUTPUTDIR = "golden_outputs"
OUTPUTS = {
    "evals": "evals.npy",
    "evecs": "evecs.npy",
    "evals_half": "evals_half.npy",
    "evecs_half": "evecs_half.npy"
}
#NOTE: Replace with your expected output file name(s). Should be in order
# of the results returned by run()

def test_example():
    example_dir = os.path.dirname(__file__)
    run = import_run(example_dir)
    name = os.path.basename(os.path.dirname(__file__))
    group = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    log_file = os.path.join(base_path, group, "status.json")

    # Load expected results
    expected = {}
    for label, fname in OUTPUTS.items():
        path = os.path.join(os.path.dirname(__file__), OUTPUTDIR, fname)
        expected[label] = np.load(path)
    
    # Get result from model
    results = run()
    if not isinstance(results, (tuple, list)):
        results = [results]

    # Prepare entry
    entry = {
        "last_pass": datetime.datetime.now().isoformat(),
        "pythtb_version": get_version("pythtb"),
        "python_version": platform.python_version(),
        "status": "pass"
    }
    
    if len(results) != len(OUTPUTS):
        entry["status"] = "fail"
        entry["reason"] = f"Expected {len(OUTPUTS)} outputs, got {len(results)}"
        raise AssertionError(entry["reason"])
    # Compare results with expected outputs
    #NOTE: Modify to match your expected output structure
    try:
        for i, (label, fname) in enumerate(OUTPUTS.items()):
            result = results[i]
            # There are degenerate states in the Haldane model, so we need to check
            # that the projectors are equivalent rather than the eigenvectors themselves.
            if label == 'evecs_half':
                P1 = result @ result.conj().T  # Projector for result
                P2 = expected[label] @ expected[label].conj().T  # Projector for golden
                assert np.allclose(P1, P2, rtol=1e-8, atol=1e-14), f"Projectors for {label} are not equivalent"
            else:
                np.testing.assert_allclose(result, expected[label], rtol=1e-8, atol=1e-14)
    except AssertionError as e:
        entry["status"] = "fail"
        entry["reason"] = f"Mismatch in {label}: {str(e)}"

    # Log the pass/fail status
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    status = {}
    if os.path.exists(log_file):
        try:
            with open(log_file) as f:
                content = f.read().strip()
                if content:
                    status = json.loads(content)
        except Exception as e:
            print(f"⚠️ Warning: Couldn't parse {log_file}, starting fresh. Reason: {e}")

    status[name] = entry
    with open(log_file, "w") as f:
        json.dump(status, f, indent=4)

    if entry["status"] == "fail":
        raise AssertionError(entry["reason"])

def get_version(pkg):
    try:
        import importlib.metadata as im
        return im.version(pkg)
    except Exception:
        return "unknown"
