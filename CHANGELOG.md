# Changelog

All notable changes to this project will be documented in this file.  
This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and follows [Semantic Versioning](https://semver.org/).

----

## [2.0.0] - 2025-06-03

### Added
- `tb_model.__repr__`: Object representation now displays `rdim`, `kdim`, and `nspin`.
- `tb_model.__str__`: Printing a `tb_model` instance now calls the `report()` (formerly `dislplay`) method.
- `tb_model.get_velocity`: Computes  $dH/dk$ (velocity operator) in the orbital basis.
- `tb_model.berry_curv`: Computes Berry curvature from $dH/dk$ elements; accepts occupied band indices.
- `tb_model.chern`: Returns Chern number for a given set of occupied bands.
- `tb_model.get_recip_lat`: Returns reciprocal lattice vectors.
- Standard files added:
    - `LICENSE`: GPL-3 license (copied over from the .txt file).
    - `CHANGELOG`: This file for tracking changes between versions. 
    - `pyproject.toml`: Now the recommended way to package Python projects.
    - `CONTRIBUTING`: Outlines expectations and guidelines for contributors.

### Changed
- **Major Refactor:** The `pythtb.py` was split into a modular package (`pythtb/` folder with submodules). This improves maintainability and readability. [See [DEVELPMENT.md](notes/DEVELOPMENT.md) for rationale and module overview.]
- `tb_model.__init__`:
    - Codebase cleaned and simplified for readability.
    - Added type hints throughout.
    - Improved error messages for clarity.
    - Added logging (using Python's `logging` module) with configurable message levels.
    - Renamed `self._nsta` to `self._nstate` to be explicit.
- `tb_model.get_orb`: Added `cartesian` boolean flag to return orbitals in Cartesian coordinates.
- `tb_model.set_onsite`:
    - Only `set` and `add` methods retained for clarity; `reset` is now merged into `set`.
      - The only difference between `set` and `reset` internally was that an exception is raised if `set` 
      is overwriting a previously set site energy. Now warns via logger if overwriting a previously set site energy.
- `tb_model.visualize`:
    - Hopping vectors depicted as curved arrows.
    - Lattice vectors shown as arrows.
    - Arrow transparency scales with hopping magnitude (max element in 2x2 matrix if spinful).
- `tb_model.display` -> `tb_model.report`:
    - Simplified and aligned output (now uses `np.array2string` with custom formatting).
    - Header is centered and capitalized.
    - Added both Cartesian and reduced units.
- `tb_model._gen_ham` -> `tb_model.get_ham`:
    - Now generates Hamiltonians for both single and multiple k-points.
    - Bottleneck code vectorized for better performance.
    - Method is now public.
- `tb_model.solve_one` / `tb_model.solve_all` -> `tb_model.solve_ham`:
    - Unified method subsumes previous methods.
    - Flag renamed: `eigvectors` → `return_eigvecs` for clarity.
    - Changed output shape: eigenvalues now indexed as `(Nk, n_state)` for vectorized workflows (matrix elements go last for NumPy linear algebra operations).
    - Eigenvectors shaped for spinful and spinless cases (see docstring for full details).
      - `n_spin`= 1: (Nk, n_state, n_state) 
      - `n_spin`= 2: (Nk, n_state, n_orb, n_spin)
      - If finite (no k axis): (n_state, ...) and spin axes are as before
    - Handles single-k-point input automatically and reproduces `solve_one`.

### Removed 
- Support for Python <3.10 ([SPEC-0](https://scientific-python.org/specs/spec-0000/))
- Deprecated `setup.py`: migration to  `pyproject.toml`.

### Deprecated
- `tb_model.solve_one`: Use `tb_model.solve_ham` instead
- `tb_model.solve_all`: Use `tb_model.solve_ham` instead
- `tb_model.display`: Use `tb_model.report` instead
- `reset` flag for `tb_model` onsite energies: Use `set` instead.

### Developer Notes
For a detailed rationale for the refactor and module breakdown, see the developer documentation [DEVELOPMENT.md](notes/DEVELOPMENT.md).

