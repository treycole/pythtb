## v2.0.0

### Improvements

#### `tb_model.__init__`
- Cleaned up and simplified the code for readability
- Using type hints to clarify data type to the user
- Some changes to error messages, being more explicit in places
- Using logger to send messages to user. Can specify level of message (warning, info, etc.). Standard module.
- Changed `self._nsta` to `self._nstate`
#### `tb_model.get_orb`
- Added `cartesian` boolean flag to return orbital vectors in cartesian units
#### `tb_model.set_onsite`
- Only keeping 'set' and 'add' for simplicity
  - using 'set' in place of 'reset' only had the difference that an exception was raised
  - A more common use case would likely be that the user intended to 'reset' anyways (from experience)
  - Now throws a warning from the logger that a site energy had already been set and will be overwritten
#### `tb_model.visualize`
- hopping vectors depicted as curved arrows
- lattice vectors shown as arrows
- legend labeling the orbitals indicated by distinct colors
- Transparency of the hopping arrow is scaled by the magnitude of the hopping.
	-  This transparency corresponds to the largest magnitude in the 2x2 matrix for spinful calculations.
#### `tb_model.display` -> `tb_model.report` 
- Simplified and shortened code
- Used `np.array2string` with custom formatter for aligning and centering
- Specify cartesian or reduced units
- Changed capitalization, centered header
#### `tb_model._gen_ham` -> `tb_model.get_ham`
- Generates Hamiltonian for both a list of k-points or a single k-point
- Source of major bottleneck in code, now vectorized for efficiency
- Promoted to a public method, likely use cases where the user wants to obtain the Hamiltonian 
#### `tb_model.solve_one` and `tb_model.solve_all` -> `tb_model.solve_ham`
- `solve_ham` subsumes `solve_one` and `solve_all`
- Flag `eigvectors` -> `return_eigvecs` for clarity
- Indexing has swapped Nk and band index
  - Returned eigenvalues have shape (Nk, n_state) instead of (n_state, Nk)
  - To vectorize diagonalization, the matrix elements need to be the last indices (Nk first) in the Hamiltonian
  - This returns the eigenstates and eigenvalues with Nk being first axis. This is a more natural indexing pattern.
- returned eigenvectors have shape 
	- spin=1: (Nk, n_state, n_state) 
	- spin=2: (Nk, n_state, n_orb, n_spin)
	- If finite, no k axis: (n_state, ...)
- Checks if k_pts is a single point, and if so, it adjusts the shape accordingly and reproduces the former `solve_one`

### Added

#### `tb_model.__repr__`
- In an interactive shell, this is what Python reports as a representation for the object.
- Shows the `rdim`, `kdim`, and `nspin`

#### `tb_model.__str__`
- String representation of `tb_model`
- `print(tb_model)` shows the former `display` (now `report`)
- `print(tb_model)` has same effect as `tb_model.report()`

#### `tb_model.get_velocity`
- computes dH/dK in the orbital basis

#### `tb_model.berry_curv`
- computes Berry curvature from dH/dk elements
- takes the occupied indices as an arguement

#### `tb_model.chern`
- returns Chern number for a set of occupied bands
- takes the occupied indices as an argument

#### `tb_model.get_recip_lat`
- returns reciprocal lattice vectors

#### Files
- `LICENSE`: copied over the .txt file to the standard `LICENSE` file
- This `CHANGELOG`: to track updates for each version
- `pyproject.toml`: this is the current recommended way to package in place of setup tools
- `CONTRIBUTING`: outlining expectations from contributors 

### Removed 
- Support for Python <3.10 (see [SPEC-0](https://scientific-python.org/specs/spec-0000/))
- `setup.py` is deprecated. Recommended to use `pyproject.toml`.
