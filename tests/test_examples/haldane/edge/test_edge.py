import os
import numpy as np
import pytest
from tests.utils import import_run

OUTPUTDIR = "golden_outputs"
# NOTE: Replace with your expected output file name(s). Should be in order
# of the results returned by run()
OUTPUTS = {
    "evals": "evals.npy",
    "evecs": "evecs.npy",
    "evals_half": "evals_half.npy",
    "evecs_half": "evecs_half.npy",
}


def _assert_eigenspaces_equal(evals, actual, expected, label):
    """Compare row-wise eigenvectors up to rotations within degenerate spaces."""
    np.testing.assert_equal(actual.shape, expected.shape)
    np.testing.assert_equal(actual.shape[0], len(evals))

    start = 0
    while start < len(evals):
        stop = start + 1
        while stop < len(evals) and np.isclose(
            evals[stop], evals[start], rtol=1e-8, atol=1e-14
        ):
            stop += 1

        # Eigenvectors are rows, so form the projector in the orbital basis.
        # Comparing the entire orthonormal basis would only compare identities.
        actual_block = actual[start:stop]
        expected_block = expected[start:stop]
        np.testing.assert_allclose(
            actual_block.T @ actual_block.conj(),
            expected_block.T @ expected_block.conj(),
            rtol=1e-8,
            atol=1e-14,
            err_msg=f"Eigenspace for {label}, states [{start}:{stop}], is not equivalent",
        )
        start = stop


def test_example():
    example_dir = os.path.dirname(__file__)
    run = import_run(example_dir)

    # Load expected results
    expected = {}
    for label, fname in OUTPUTS.items():
        path = os.path.join(os.path.dirname(__file__), OUTPUTDIR, fname)
        expected[label] = np.load(path)

    # Get result from model
    results = run()
    if not isinstance(results, (tuple, list)):
        results = [results]
    if len(results) != len(OUTPUTS):
        raise AssertionError(f"Expected {len(OUTPUTS)} outputs, got {len(results)}")
    for label, result in zip(OUTPUTS, results):
        if label.startswith("evecs"):
            _assert_eigenspaces_equal(
                expected[label.replace("evecs", "evals", 1)],
                result,
                expected[label],
                label,
            )
        else:
            np.testing.assert_allclose(result, expected[label], rtol=1e-8, atol=1e-14)


@pytest.fixture
def eigenpairs():
    evals = np.array([0.0, 0.0, 1.0])
    evecs = np.array([[1, 1j, 0], [1j, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
    return evals, evecs


@pytest.mark.parametrize("change", ["phase", "degenerate_rotation", "degenerate_swap"])
def test_equivalent_eigenspaces(eigenpairs, change):
    evals, expected = eigenpairs
    actual = expected.copy()
    if change == "phase":
        actual *= np.exp(1j * np.array([0.3, -1.2, np.pi]))[:, None]
    elif change == "degenerate_rotation":
        rotation = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
        actual[:2] = rotation @ actual[:2]
    else:
        actual[[0, 1]] = actual[[1, 0]]

    _assert_eigenspaces_equal(evals, actual, expected, "evecs")


@pytest.mark.parametrize("change", ["different_energy_swap", "orbital_swap", "norm"])
def test_inequivalent_eigenspaces(eigenpairs, change):
    evals, expected = eigenpairs
    actual = expected.copy()
    if change == "different_energy_swap":
        actual[[0, 2]] = actual[[2, 0]]
    elif change == "orbital_swap":
        actual[:, [0, 2]] = actual[:, [2, 0]]
    else:
        actual[0] *= 2

    with pytest.raises(AssertionError, match="Eigenspace for evecs"):
        _assert_eigenspaces_equal(evals, actual, expected, "evecs")
