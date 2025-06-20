import numpy as np
from pythtb import TBModel
import pytest

@pytest.mark.parametrize("dim_k, dim_r, lat_vecs, orbital_pos, nspin", [
    (1, 1, [[1]], [[0]], 1),  # 1D
    (1, 1, [[1]], [[0], [0.5]], 1),  # 1D with two orbitals
    (2, 2, [[1, 0], [0, 1]], [[0, 0], [0.5, 0.5]], 1),  # 2D
    (1, 2, [[1, 0], [0, 1]], [[0, 0], [0.5, 0.5]], 1),  # 1D k-space, 2D real space
    (3, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[0, 0, 0], [0.25, 0.25, 0.25]], 1),  # 3D
    (2, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[0, 0, 0], [0.25, 0.25, 0.25]], 1),  # 2D k-space, 3D real space
    (1, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[0, 0, 0], [0.25, 0.25, 0.25]], 1),  # 1D k-space, 3D real space
    (1, 1, [[1]], [[0]], 2),  # 1D with nspin=2
    (1, 1, [[1]], [[0], [0.5]], 2),  # 1D with two orbitals and nspin=2
    (2, 2, [[1, 0], [0, 1]], [[0, 0], [0.5, 0.5]], 2),  # 2D with nspin=2
    (1, 2, [[1, 0], [0, 1]], [[0, 0], [0.5, 0.5]], 2),  # 1D k-space, 2D real space with nspin=2
    (3, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[0, 0, 0], [0.25, 0.25, 0.25]], 2),  # 3D with nspin=2
    (2, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[0, 0, 0], [0.25, 0.25, 0.25]], 2),  # 2D k-space, 3D real space with nspin=2
    (1, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[0, 0, 0], [0.25, 0.25, 0.25]], 2),  # 1D k-space, 3D real space with nspin=2
])
def test_tbmodel_initialization(dim_k, dim_r, lat_vecs, orbital_pos, nspin):
    """
    Test the TBModel initialization with various dimensions and lattice vectors.
    """
    # Create a TBModel instance
    test_model = TBModel(dim_k, dim_r, lat=lat_vecs, orb=orbital_pos, nspin=nspin)

    # Check if the dimensions are set correctly
    assert test_model.dim_k == dim_k, f"dim_k should be {dim_k}"
    assert test_model.dim_r == dim_r, f"dim_r should be {dim_r}"
    assert test_model.nspin == nspin, f"nspin should be {nspin}"
    
    # Check if the lattice vectors and orbital positions are set correctly
    np.testing.assert_array_equal(test_model.lat_vecs, lat_vecs, "lat_vecs should match")
    np.testing.assert_array_equal(test_model.orb_vecs, orbital_pos, "orbital positions should match")
    
    # Check if the number of orbitals is correct
    assert test_model.norb == len(orbital_pos), "norb should match the number of orbital positions"

@pytest.mark.parametrize("dim_k, dim_r, lat_vecs, nspin", [
    (3, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 1),
    (3, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 2),
])
def test_bravais_lattice(dim_k, dim_r, lat_vecs, nspin):
    """
    Test the TBModel's bravais_lattice method.
    """
    model = TBModel(dim_k, dim_r, lat=lat_vecs, orb="bravais", nspin=nspin)

    assert model.norb == 1, "Bravais lattice should have one orbital"
    orb_vec = np.zeros((1, dim_r))
    np.testing.assert_array_equal(model.orb_vecs, orb_vec, "Bravais lattice orbital position should be zero vector")

@pytest.mark.parametrize("dim_k, dim_r, lat_vecs, orbital_pos, nspin", [
    (3, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0, 1),  # 3D
    (2, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 5, 1),  # 2D k-space, 3D real space
    (1, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 4, 1),  # 1D k-space, 3D real space
    (3, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0, 2),  # 3D with nspin=2
    (2, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 10, 2),  # 2D k-space, 3D real space with nspin=2
    (1, 3, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 1, 2),  # 1D k-space, 3D real space with nspin=2
    (2, 2, [[1, 0], [0, 1]], 1, 1),  # 2D with nspin=1
    (1, 2, [[1, 0], [0, 1]], 2, 1),  # 1D k-space, 2D real space with nspin=1
    (3, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 4, 1),  # Simple cubic lattice
    (2, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 2, 1),  # Simple cubic lattice in k-space
])
def test_origin_orbs(dim_k, dim_r, lat_vecs, orbital_pos, nspin):
    """
    Test the TBModel's origin_orbs method.
    """
    model = TBModel(dim_k, dim_r, lat=lat_vecs, orb=orbital_pos, nspin=nspin)

    assert model.norb == orbital_pos, "norb should match the number of orbital positions"
    # assert that lat_vecs are all at origin
    np.testing.assert_array_equal(model.orb_vecs, np.zeros_like(model.orb_vecs),
                                  "lattice vectors should be at the origin")
    assert model.orb_vecs.shape[0] == orbital_pos, "lat_vecs should have the same number of vectors as orbitals"