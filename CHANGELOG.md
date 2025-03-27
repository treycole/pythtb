## [2.0.0]

### Improvements

#### `tb_model.__init__`
- Cleaned up and simplified the code for readability
- Using type hints to clarify data type to the user
- Some changes to error messages, being more explicit in places
- Using logger to send messages to user. Can specify level of message (warning, info, etc.). Standard module.
- Changed `self._nsta` to `self._nstate`

#### `tb_model.__repr__`
- In an interactive shell, this is what Python reports as a representation for the object.
- Shows the `rdim`, `kdim`, and `nspin`

#### `tb_model.__str__`
- String representation of `tb_model`
- `print(tb_model)` shows the former `display` (now `report`)
- `print(tb_model)` has same effect as `tb_model.report()`
#### `tb_model.get_orb`
- Added `cartesian` boolean flag to return orbital vectors in cartesian units
#### `tb_model.display` -> `tb_model.report` 
- Simplified and shortened code
- Used `np.array2string` with custom formatter for aligning and centering
- Specify cartesian or reduced units
- Changed capitalization, centered header
#### `tb_model.set_onsite`
- Only keeping 'set' and 'add' for simplicity
#### `tb_model.visualize`
- Enhanced visualization, with hopping vectors showing as curved arrows, lattice vectors being arrows, and a legend indicating orbitals indicated by distinct colors
- Transparency of the hopping arrow is scaled by the magnitude of the hopping. For spinful calculations, this transparency corresponds to the largest magnitude in the 2x2 matrix.
#### `tb_model._gen_ham` -> `tb_model.get_ham`
- Generates Hamiltonian for a list of k-points, rather than a single k-point
- Source of major bottleneck in code
- Promoted to public method, likely use cases where user wants to obtain the Hamiltonian 
#### `tb_model.solve_one` and `tb_model.solve_all` -> `tb_model.solve_ham`
- `solve_ham` subsumes `solve_one` and `solve_all`
- Flag `eigvectors` -> `return_eigvecs`
- Indexing has swapped Nk and band index
  - returned eigenvalues have shape (Nk, n_state) instead of (n_state, Nk)
  - Allows for vectorizing the diagonalization
- returned eigenvectors have shape 
	- spin=1: (Nk, n_state, n_state) 
	- spin=2: (Nk, n_state, n_orb, n_spin)
	- If finite, no k axis: (n_state, ...)
- Checks if k_pts is a single point, and if so, it adjusts the shape accordingly and reproduces the former `solve_one`

### Added
- `LICENSE`
- This `CHANGELOG`
- `pyproject.toml`
- `CONTRIBUTING`

### Removed 
- Support for Python <3.10 (see [SPEC-0](https://scientific-python.org/specs/spec-0000/))
