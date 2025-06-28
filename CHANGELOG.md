# Changelog

All notable changes to this project will be documented in this file.  
This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and follows [Semantic Versioning](https://semver.org/).

------

## [2.0.0] - 2025-06-03
---
### Changed

- Renamed public classes (see [DEVLOG](notes/DEVLOG.md) for more details)
    - `tb_model` -> `TBModel`
    - `wf_array` -> `WFArray`
    - `w90` -> `W90`

- Examples are now grouped categorically into folders

##### `TBModel`

- `TBModel.get_orb`:
    - Added `cartesian` boolean flag to return orbitals in Cartesian coordinates. It is `False` by default.
- `TBModel.set_onsite`:
    - Only `set` and `add` parameters retained; `reset` is now merged into `set`.
- `TBModel.visualize` has been improved
    - Hopping vectors depicted as curved arrows instead of two straight lines at an angle. 
    - Lattice vectors shown as arrows, unit cell delineated with dotted lines. 
    - Arrow transparency scales with hopping magnitude (max element in 2x2 matrix if spinful) to give an idea of the strength of the hopping terms in the model.
- `tb_model.display` -> `TBModel.report`:
    - Simplified and aligned output (now uses `np.array2string` with custom formatting).
    - Header is centered and capitalized.
    - Prints orbital vectors in both Cartesian and reduced units.
- `tb_model.solve_one` / `tb_model.solve_all` -> `TBModel.solve_ham`:
    - Unified method subsumes previous methods.
    - Flag renamed: `eigvectors` -> `return_eigvecs` for clarity.
    - Changed output shape: eigenvalues now indexed as `(Nk, n_state)` for vectorized workflows (matrix elements go last for NumPy linear algebra operations).
    - Eigenvectors shaped for spinful and spinless cases (see docstring for full details).
      - `n_spin`= 1: (Nk, n_state, n_state) 
      - `n_spin`= 2: (Nk, n_state, n_orb, n_spin)
      - If finite (no k axis): (n_state, ...) and spin axes are as before
    - Handles single-k-point input automatically and reproduces `solve_one`.

##### `WFArray`
- `WFArray.berry_flux`
    - Flag renamed: `occ` -> `state_idx`
    - Flag renamed: `dirs` -> `plane`
    - Flag removed: `individual_phases`
        - This flag previously returned the integrated Berry flux in the plane as a function of the remaining parameters. For clarity, it is now up to the user to sum over the proper axes if they want to integrate the Berry flux. The Berry flux will have axes for all parameter directions. 
    - Default behavior change: when `dirs` is unspecified (or `None`) the returned Berry flux will have 2 additional axes (first and second) over all combinations of planes (e.g. berry_flux()[0,1] is the Berry flux in the 0,1 plane)
    - Substantial speed improvements using NumPy vectorization

#### `W90`
- `W90.w90_bands_consistency`
    - Returned energies now have shape (kpts, band) instead of (band, kpts). This matches the shape of the returned eigenvectors in `TBModel.solve_ham`.

---

### Added

##### Models
- Added a [folder of example models](pythtb/models) that is importable using, e.g.,
  ```
  from pythtb.models import haldane
  my_model = haldane(delta, t, t2)
  ```
##### Examples
- [visualize_3d.py](examples/visualize/visualize_3d.py): demonstrates the 3D visualization feature for `TBModel`
- [ssh.py](examples/ssh/ssh.py): Constructs the ssh model and plots the band structure with a slider to change the intracell hopping. 

##### `TBModel`
- `TBModel.__repr__`: Object representation now displays `rdim`, `kdim`, and `nspin`. 
- `TBModel.__str__`: Printing a `TBModel` instance using `print(TBModel)` calls the `report()` (formerly `dislplay`) method and prints model information.
- `TBModel.get_ham`: Generates Hamiltonians for both single and multiple k-points.
- `TBModel.get_velocity`: Computes $dH/dk$ (velocity operator) in the orbital basis.
- `TBModel.berry_curv`: Computes Berry curvature from $dH/dk$ elements using the Kubo formula. Accepts occupied band indices. Assumes there is a global gap that defines the occupied and unoccupied bands. 
- `TBModel.chern`: Returns Chern number for a given set of occupied bands using the Berry curvature from above. Assumes there is a global gap that defines the occupied and unoccupied bands. 
-  `TBModel.visualize3d`: For 3-dimensional tight-binding models, displays a 3d figure of the tight-binding orbitals using `plotly`. Also prints a legend with the model terms (onsite energies, orbital positions). The figure can be rotated and zoomed in. When highlighting an orbital or bond, the onsite or hopping terms appear. 
- `TBModel.get_recip_lat`: Returns reciprocal lattice vectors.
- Added read-only retrieval of core TBModel attributes (e.g., `dim_r`, `dim_k`, `nspin`, `per`, `norb`, `nstate`, `lat`, `orb`, `site_energies`, and `hoppings`) using e.g. `my_model.dim_r` preventing unintended modification of internal model parameters.

##### `WFArray`
- `WFArray.chern_num`
    - Returns the chern number for a given plane
- `WFArray.wilson_loop`
    - Computes the Wilson loop unitary matrix for a loop of states
- `WFArray.get_links`
    - Computes the unitary part of the overlap between the mesh of states and their nearest neighbors in each direction
- `WFArray.solve_on_path`
    - Populates a 1D `WFArray` with states diagonalized on a 1D k-path
- `WFArray.get_projectors`
    - Returns the band projectors and optionally their compliment 
- `WFArray.get_bloch_states`
    - When the states populated are all Bloch states (defined on k-mesh), this function applies the $e^{i \mathbf{k} \cdot \mathbf{r}}$ phase factors and returns both the cell-periodic $u_{n\mathbf{k}}$ and the Bloch states $\phi_{n\mathbf{k}}$.
- `WFArray.get_states`
    - Returns the `WFArray` states in NumPy form. 
    - Has an optional flag to flatten the spin axis in cases where the states are spinful
- Properties
    - Added a series of read-only properties for certain attributes
--- 

### Removed 
- Support for Python <3.10 ([SPEC-0](https://scientific-python.org/specs/spec-0000/))
- Deprecated `setup.py`: migration to `pyproject.toml`.
- `WFArray.berry_flux` flag `individual_phases`
- `TBModel.solve_one`
- `TBModel.solve_all`

---

### Deprecated
- `tb_model.solve_one`: Use `TBModel.solve_ham` instead
- `tb_model.solve_all`: Use `TBModel.solve_ham` instead
- `tb_model.display`: Use `TBModel.report` instead
- `reset` flag for `TBModel.set_onsite`: Use `set` instead.

---

### Developer Notes
For a detailed explanation of the changes, see the developer documentation [DEVELOPMENT.md](notes/DEVELOPMENT.md).

