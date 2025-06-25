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

- Examples are now separated categorically into folders

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
    - Flag renamed: `eigvectors` → `return_eigvecs` for clarity.
    - Changed output shape: eigenvalues now indexed as `(Nk, n_state)` for vectorized workflows (matrix elements go last for NumPy linear algebra operations).
    - Eigenvectors shaped for spinful and spinless cases (see docstring for full details).
      - `n_spin`= 1: (Nk, n_state, n_state) 
      - `n_spin`= 2: (Nk, n_state, n_orb, n_spin)
      - If finite (no k axis): (n_state, ...) and spin axes are as before
    - Handles single-k-point input automatically and reproduces `solve_one`.

##### `WFArray`


---

### Added

##### Models
- Added a [folder of example models](pythtb/models) that is importable using, e.g.,
  ```
  from pythtb.models import haldane
  my_model = haldane(delta, t, t2)
  ```
##### Examples
- Added an example, [visualize_3d.py](examples/visualize/visualize_3d.py), for the 3D visualization feature 

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

--- 

### Removed 
- Support for Python <3.10 ([SPEC-0](https://scientific-python.org/specs/spec-0000/))
- Deprecated `setup.py`: migration to `pyproject.toml`.

---

### Deprecated
- `tb_model.solve_one`: Use `TBModel.solve_ham` instead
- `tb_model.solve_all`: Use `TBModel.solve_ham` instead
- `tb_model.display`: Use `TBModel.report` instead
- `reset` flag for `TBModel.set_onsite`: Use `set` instead.

---

### Developer Notes
For a detailed explanation of the changes, see the developer documentation [DEVELOPMENT.md](notes/DEVELOPMENT.md).

