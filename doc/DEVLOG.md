# Development Notes for PythTB

This file contains as-needed explanations for design decisions, rationales, and best-practices references behind each major version update. It is intended for contributors, maintainers, and advanced users interested in understanding *why* the code is structured as it is.

## v2.0.0 – Major Updates & API Modernization

These changes were made to improve the long-term health of the codebase, facilitate contributions, and align with modern Python best practices. Below is a summary of the changes, and the rationale behind them.

### Table of Contents

1. [Project Structure and Modularization](#1-project-structure-and-modularization)
2. [Class Renaming and Python 3 Modernization](#2-class-renaming-and-python-3-modernization)
3. [API Refactor: Merging, Renaming, and Removing](#3-api-refactor-merging-renaming-and-removing)
4. [Properties and Attribute Encapsulation](#4-properties-and-attribute-encapsulation)
5. [Type Hints and Static Analysis](#5-type-hints-and-static-analysis)
6. [Error Handling Improvements](#6-error-handling-improvements)
7. [Logging and Messaging](#7-logging-and-messaging)
8. [Vectorization and Performance](#8-vectorization-and-performance)
9. [Internal Structure: Utilities, Plotting, and Helpers](#9-internal-structure-utilities-plotting-and-helpers)
10. [Packaging, Distribution, and Standards](#10-packaging-distribution-and-standards)
11. [Python Version Support](#11-python-version-support)
12. [References and Further Reading](#12-references-and-further-reading)


### 1. Project Structure and Modularization

**What Changed:**
- The former monolithic `pythtb.py` file (4000+ lines) was split into a true Python package in the `pythtb/` directory with smaller, purpose-driven modules:
    - `tb_model.py` (main tight-binding model class)
    - `wf_array.py` (wavefunction storage and post-processing)
    - `k_mesh.py` (k-point mesh construction and related routines)
    - `plotting.py` (plotting utilities, not part of public API)
    - `utils.py` (general-purpose helpers)

**Rationale:**
- Smaller files encapsulate related logic, improving maintainability, readability, and onboarding for new contributors.
- Encourages separation of concerns, easier unit testing, and refactoring.
- Aligns with [Python Packaging Authority best practices](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
- See also: [Martin Fowler, "Refactoring"](https://martinfowler.com/books/refactoring.html), [Real Python: Python Application Layouts](https://realpython.com/python-application-layouts/)


### 2. Class Renaming 

**What Changed:**
- Renamed classes to **CapWords** (PascalCase) style per [PEP 8](https://peps.python.org/pep-0008/#class-names) convention:
    - `tb_model` → `TBModel`
    - `wf_array` → `WFArray`
    - `w90` → `W90`
- Removed explicit inheritance from `object` (i.e., `class TBModel(object):` is now `class TBModel:`).

**Rationale:**
- Follows modern naming conventions and community expectations.
- Makes it easier for users and contributors to identify class objects at a glance.
- In Python 3, all classes are new-style by default; `object` base is redundant.
ty standards.

### 3. API Refactor: Merging, Renaming, and Removing 

**What Changed:**

- Merged redundant `tb_model.solve_one` and `tb_model.solve_all` into `TBModel.solve_ham`, which now handles both single and multiple k-points via input type. 
    - In cases where `solve_one` would have been used, `solve_ham` reproduces `solve_one` when a single k-point is passed. This returns the eignvalues (and eigenvectors) without an additional axis for k-points. 
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
 - Removed the `reset` flag from `set_onsite`.
    - If the onsite energy has already been set, a warning is sent to the user via the logger and the function overwrites the value. 
    - The only internal difference between `set` and `reset` is that if `set` is used to overwrite an onsite energy, an error is thrown. 
- Renamed boolean flag for returning eigenvectors in `solve_ham`: `eigvectors` → `return_eigvecs` for clarity.

**Rationale:**
- Explicit is better than implicit ([Zen of Python, PEP 20](https://peps.python.org/pep-0020/)).
- Avoid API bloat: Reducing duplicate or unnecessary methods lowers cognitive overhead.
- Predictable naming conventions make the library easier to use and less error-prone.

**References:**
- [PEP 8 – Naming Conventions](https://peps.python.org/pep-0008/#naming-conventions)
- [API Design Guidelines](https://docs.python-guide.org/writing/style/#api-design)

### 4. Type Hints and Error Handling

**Summary:**  
Type annotations were added throughout the codebase, and error messages were improved for clarity.

E.g. instead of `raise Error`, use `raise TypeError` to specify that the type of the object is incorrect.

**Rationale:**
- **Static analysis:** Type hints enable better linting, static analysis, and IDE support.
- **User guidance:** Clear errors help users debug issues without digging into internals.
- **Modern best practice:** Type hints are now standard for modern Python libraries.

**References:**
- [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- [Type Checking – Python Docs](https://docs.python.org/3/library/typing.html)
- [Effective Error Messages](https://nedbatchelder.com/text/errors.html)


### 5. Logging vs. Print Statements

**Summary:**  
Replaced print-based messaging with Python’s `logging` module.

**Rationale:**
- **Granularity:** Log messages can be filtered by severity (debug, info, warning, error).
- **Configurability:** Users can control where log messages go (stdout, file, etc.).
- **Professional standards:** Print statements are for scripts; logging is for libraries.

**References:**
- [Logging HOWTO – Python Docs](https://docs.python.org/3/howto/logging.html)
- [Real Python: Logging in Python](https://realpython.com/python-logging/)

### 6. Vectorization and Performance

**What Changed:**  
Performance bottlenecks were addressed via NumPy vectorization. For example, the `tb_model._gen_ham` and `tb_model.solve_all` functions for generating and diagonalizing the Hamiltonian were a major bottleneck. It has been restructured to employ NumPy vectorization, with no explicit `for` loops over the k-points.

**Rationale:**
- **Efficiency:** Loops in Python are slow; NumPy operations are orders of magnitude faster ([Why NumPy is Fast](https://numpy.org/doc/stable/user/whatisnumpy.html#why-is-numpy-fast)).
- **Scaling:** Enables calculations over large k-point grids and many bands.
- **Natural indexing:** Output shapes now follow the most common convention (k-points as the leading axis).

**References:**
- [NumPy Best Practices](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [SciPy Lecture Notes: Performance](https://scipy-lectures.org/advanced/optimizing/)

### 7. Output and Display Refactor

**Summary:**  
Display functionality (`display`) was renamed and refactored as `report` for clarity. Uses `np.array2string` for aligned output.

**Rationale:**
- **Clarity:** Method name reflects its role (reporting summary information).
- **Customizability:** Supports formatting (centering).
- **Consistency:** Aligns with Python dunder methods (`__str__`, `__repr__`).

**References:**
- [PEP 257 – Docstring Conventions](https://peps.python.org/pep-0257/)
- [Python Data Model (dunder methods)](https://docs.python.org/3/reference/datamodel.html#special-method-names)

### 8. Packaging and Distribution

**Summary:**  
Transitioned from `setup.py` to `pyproject.toml`, added standard files (`LICENSE`, `CHANGELOG`, `CONTRIBUTING`).
    - `LICENSE`: GPL-3 license (copied over from the .txt file).
    - `CHANGELOG`: This file for tracking changes between versions. 
    - `pyproject.toml`: Now the recommended way to package Python projects.
    - `CONTRIBUTING`: Outlines expectations and guidelines for contributors.

**Rationale:**
- **PEP 517/518 compliance:** `pyproject.toml` is the new standard for Python packaging.
- **Reproducibility and interoperability:** Tools like Poetry, Flit, and pip rely on standardized metadata.
- **Contributor clarity:** `CONTRIBUTING` and `LICENSE` are now expected in open source.

**References:**
- [Packaging Python Projects – Official Tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [PEP 517 – pyproject.toml build-system](https://peps.python.org/pep-0517/)
- [GitHub Community Standards](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/)

### 9. Python Version Support

**Summary:**  
Dropped support for Python <3.10. Now require Python 3.10+.

**Rationale:**
- **Simplifies code:** Allows use of latest syntax and features (match-case, improved typing, etc.).
- **Upstream dependencies:** Many scientific libraries have dropped older versions.
- **Follows [SPEC-0](https://scientific-python.org/specs/spec-0000/):** Scientific Python minimum standards.

**References:**
- [SPEC-0000](https://scientific-python.org/specs/spec-0000/)
- [Python Release Calendar](https://devguide.python.org/versions/)


### 10. References and Further Reading

- [The Zen of Python](https://peps.python.org/pep-0020/)
- [Effective Python, 2nd Ed. (Brett Slatkin)](https://effectivepython.com/)
- [Refactoring (Martin Fowler)](https://martinfowler.com/books/refactoring.html)
- [Python in a Nutshell (Martelli et al.)](https://www.oreilly.com/library/view/python-in-a/0596100469/)

**If you have questions or want to propose a change, please open an issue or start a discussion.**  
For contributing guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).