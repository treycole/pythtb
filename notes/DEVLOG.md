# Development Notes for PythTB

This file contains explains the design decisions and rationales behind each major version update. It is intended for contributors, maintainers, and advanced users interested in understanding *why* the code is structured as it is. 

## v2.0.0 – Major Updates & API Modernization

These changes were made to improve the long-term health of the codebase, facilitate contributions, and align with modern Python practices. Below is an overview of the thematic changes of the package.

Below is a summary

### Table of Contents

1. [Project Structure and Modularization](#1-project-structure-and-modularization)
2. [Class Renaming and Python 3 Modernization](#2-class-renaming)
3. [API Refactor: Merging, Renaming, and Removing](#3-api-refactor-merging-renaming-and-removing)
4. [Vectorization and Performance](#4-vectorization-and-performance)
5. [Type Hints and Error Handling](#5-type-hints-and-error-handling)
6. [Logging & Messaging](#6-logging-vs-print-statements)
7. [Packaging and Distribution](#7-packaging-and-distribution)
8. [Python Version Support](#8-python-version-support)
9. [Testing and Continuous Integration](#9-testing-and-continuous-integration)

### 1. Project Structure and Modularization

**What Changed:**
- Split the monolithic `pythtb.py` (4000+ lines) into a package in the `pythtb/` directory with smaller, purpose-driven modules:
    - `tb_model.py` (tight-binding model class - `TBModel`)
    - `wf_array.py` (wavefunction array - `WFArray`)
    - `k_mesh.py` (k-point mesh construction and related routines)
    - `plotting.py` (plotting utilities, not part of public API)
    - `utils.py` (general-purpose helpers)

**Rationale:**
- Encapsulates related logic for better maintainability and readability
- Facilitates separation of concerns, unit testing, and refactoring in the future
- Aligns with [Python Packaging Authority best practices](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
- See also: [Real Python: Python Application Layouts](https://realpython.com/python-application-layouts/)

### 2. Class Renaming 

**What Changed:**
- Renamed classes to **CapWords** (PascalCase) style per [PEP 8](https://peps.python.org/pep-0008/#class-names) convention:
    - `tb_model` → `TBModel`
    - `wf_array` → `WFArray`
    - `w90` → `W90`
- Dropped redundant `object` base class (i.e., `class TBModel(object):` is now `class TBModel:`) - all classes are new-style in Python 3.

**Rationale:**
- Follows community conventions.
- Improves distinguishability of classes and functions.

### 3. API Refactor: Merging, Renaming, and Removing 

**What Changed:**

_Example_
- Merged `tb_model.solve_one` and `tb_model.solve_all` into `TBModel.solve_ham`
    - In cases where `solve_one` would have been used, `solve_ham` reproduces `solve_one` if a single k-point is passed. This returns the eigenvalues (and eigenvectors) without an additional axis for k-points. 
    - Example:
        - Before:  
        ```python
        kpt = [0, 0]
        kpts = [[0,0], [0, 0.5], [0.5, 0.5], [0.5, 0]]
        evals = model.solve_one(kpt) # shape: (n_state)
        evals_all = model.solve_all(kpts) # shape: (4, n_state)
        ```
        - After:  
        ```python
        kpt = [0, 0]
        kpts = [[0,0], [0, 0.5], [0.5, 0.5], [0.5, 0]]
        evals = model.solve_ham(kpt)  # shape: (n_state)
        evals_all = model.solve_ham(kpts) # shape: (4, n_state)
        ```
     - Removed the `reset` flag from `set_onsite`; overwrites now log a warning instead of requiring a flag
        - The only internal difference between `set` and `reset` is if `set` is used to overwrite an onsite energy, an error is thrown. 
    - Renamed boolean flag for returning eigenvectors in `solve_ham`: `eigvectors` → `return_eigvecs` for clarity.

**Rationale:**
- Reducing duplicate methods lowers cognitive load.
- Predictable, explicit arguments make the library easier to use and less error-prone ([Zen of Python, PEP 20](https://peps.python.org/pep-0020/)).
- [API Design Guidelines](https://docs.python-guide.org/writing/style/#api-design)

### 4. Vectorization and Performance

**What Changed:**  
- Performance bottlenecks were addressed via NumPy vectorization.
- For example, the `tb_model._gen_ham` and `tb_model._sol_ham` functions for generating and diagonalizing the Hamiltonian were a major bottleneck. It has been restructured to employ NumPy vectorization, with no explicit `for` loops over the k-points.

**Rationale:**
- Loops in Python are slow; NumPy operations are orders of magnitude faster ([Why NumPy is Fast](https://numpy.org/doc/stable/user/whatisnumpy.html#why-is-numpy-fast)).
- Enables calculations over large k-point grids and many bands.
- Output shapes now follow the most common convention (k-points as the leading axis).

**References:**
- [NumPy Best Practices](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [SciPy Lecture Notes: Performance](https://scipy-lectures.org/advanced/optimizing/)


### 5. Type Hints and Error Handling

**What Changed:** 
- Added type annotations throughout codebase ([PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- Replaced generic `raise Error` with specific exceptions, e.g., `TypeError` or `ValueError` 

**Rationale:**
- Type hints enable better linting and IDE support. ([Type Checking – Python Docs](https://docs.python.org/3/library/typing.html))
- Clear errors help users debug issues without digging into internals ([Effective Error Messages](https://nedbatchelder.com/text/errors.html)

### 6. Logging vs. Print Statements

**What Changed:**  
- Replaced print-based messaging with Python’s `logging` module.

**Rationale:**
- Log messages can be filtered by severity (DEBUG, INFO, WARNING, ERROR).
- Users can control where log messages go (stdout, file, etc.).
- [Logging HOWTO – Python Docs](https://docs.python.org/3/howto/logging.html)
- [Real Python: Logging in Python](https://realpython.com/python-logging/)


### 7. Packaging and Distribution

**What Changed:**  
Transitioned from `setup.py` to `pyproject.toml`, added standard files (`LICENSE`, `CHANGELOG`, `CONTRIBUTING`).
    - `LICENSE`: GPL-3 license (copied over from the .txt file).
    - `CHANGELOG`: This file for tracking changes between versions. 
    - `pyproject.toml`: Now the recommended way to package Python projects.
    - `CONTRIBUTING`: Outlines expectations and guidelines for contributors.

**Rationale:**
- `pyproject.toml` is the new standard for Python packaging.
- Tools like Poetry, Flit, and pip rely on standardized metadata.
-  `CONTRIBUTING` and `LICENSE` are now expected in open source.
- [Packaging Python Projects – Official Tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [PEP 517 – pyproject.toml build-system](https://peps.python.org/pep-0517/)
- [GitHub Community Standards](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/)

### 8. Python Version Support

**What Changed:**  
- Dropped support for Python <3.10. Now require Python 3.10+.

**Rationale:**
- Allows use of latest syntax and features (match-case, improved typing, etc.).
- Many scientific libraries have support fordropped older versions.
- Follows [SPEC-0](https://scientific-python.org/specs/spec-0000/): Scientific Python minimum standards.

### 9. Testing and Continuous Integration

**What Changed:**
- Added a [tests/test_examples](tests/test_examples) directory with pytest-based regression tests for all examples
- Each test runs the example scripts and compares outputs to reference data from v1.8.
- Added a [tests/test_examples/make_test_example.py](tests/test_examples/make_test_example.py) to automatically generate a skeleton for new tests to be made in the future.
- Added a [tests/report_test_status.py](tests/test_examples/make_test_example.py) to print a summary of which tests are passing and the date/time of last pass/fail.
- See [tests/test_examples/README.md](tests/test_examples/README.md) for more information.

**Rationale:**
- Ensures backward compatibility and prevents regressions during refactors or feature additions
- Encourages contributors to include tests alongside new code


**If you have questions or want to propose a change, please open an issue or start a discussion.**  
For contributing guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
