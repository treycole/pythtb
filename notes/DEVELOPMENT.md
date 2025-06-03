# Development Notes for PythTB

This file contains as-needed explanations for design decisions, rationales, and best-practices references behind each major version update. 
It is intended for contributors, maintainers, and advanced users interested in understanding *why* the code is structured as it is.

## v2.0.0

### Table of Contents

1. [Project Structure and Modularization](#project-structure-and-modularization)
2. [API Changes and Naming](#api-changes-and-naming)
3. [Type Hints and Error Handling](#type-hints-and-error-handling)
4. [Logging vs. Print Statements](#logging-vs-print-statements)
5. [Vectorization and Performance](#vectorization-and-performance)
6. [Output and Display Refactor](#output-and-display-refactor)
7. [Packaging and Distribution](#packaging-and-distribution)
8. [Python Version Support](#python-version-support)
9. [References and Further Reading](#references-and-further-reading)

### 1. Project Structure and Modularization

**Summary:**  
Previously, the project was organized as a single large module (`pythtb.py`). In v2.0.0, this was refactored into a **package directory** (`pythtb/`) with smaller, purpose-driven modules (e.g., `k_mesh.py`, `plotting.py`, `model.py`).

**Rationale:**
- **Separation of concerns:** Smaller files encapsulate related logic, making code easier to navigate, test, and reuse.
- **Avoid "God" files:** Large monolithic modules become unmaintainable ([Martin Fowler, Refactoring](https://martinfowler.com/books/refactoring.html)).
- **Easier onboarding:** New contributors can focus on a subset of functionality.
- **Testability:** Each module can have targeted unit tests.

**References:**
- [Python Packaging Authority: Structuring Your Project](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Real Python: Python Application Layouts](https://realpython.com/python-application-layouts/)

**Structure Example:**
- pythtb
    - __init__.py
    - plotting.py
    - k_mesh.py
    - tb_model.py
    - wf_array.py
    ...
- tests
    - test_plotting.py
    - test_tbmodel.py
    ...

### 2. API Changes and Naming

**Summary:**  
The API was simplified and made more explicit. Examples:
- Combined `solve_one` and `solve_all` into `solve_ham`.
- Changed ambiguous parameter names (e.g., `eigvectors` -> `return_eigvecs`).
- Deprecated redundant methods (`reset` merged into `set`).

**Rationale:**
- **Explicit is better than implicit** ([Zen of Python, PEP 20](https://peps.python.org/pep-0020/)).
- **Avoid API bloat:** Reducing duplicate or unnecessary methods lowers cognitive overhead.
- **Naming consistency:** Predictable APIs make the library easier to use and less error-prone.

**References:**
- [PEP 8 – Naming Conventions](https://peps.python.org/pep-0008/#naming-conventions)
- [API Design Guidelines](https://docs.python-guide.org/writing/style/#api-design)


### 3. Type Hints and Error Handling

**Summary:**  
Type annotations were added throughout the codebase, and error messages were improved for clarity.

**Rationale:**
- **Static analysis:** Type hints enable better linting, static analysis, and IDE support.
- **User guidance:** Clear errors help users debug issues without digging into internals.
- **Modern best practice:** Type hints are now standard for modern Python libraries.

**References:**
- [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- [Type Checking – Python Docs](https://docs.python.org/3/library/typing.html)
- [Effective Error Messages](https://nedbatchelder.com/text/errors.html)


### 4. Logging vs. Print Statements

**Summary:**  
Replaced print-based messaging with Python’s `logging` module.

**Rationale:**
- **Granularity:** Log messages can be filtered by severity (debug, info, warning, error).
- **Configurability:** Users can control where log messages go (stdout, file, etc.).
- **Professional standards:** Print statements are for scripts; logging is for libraries.

**References:**
- [Logging HOWTO – Python Docs](https://docs.python.org/3/howto/logging.html)
- [Real Python: Logging in Python](https://realpython.com/python-logging/)

### 5. Vectorization and Performance

**Summary:**  
Performance bottlenecks (notably Hamiltonian generation and diagonalization) were addressed via NumPy vectorization.

**Rationale:**
- **Efficiency:** Loops in Python are slow; NumPy operations are orders of magnitude faster ([Why NumPy is Fast](https://numpy.org/doc/stable/user/whatisnumpy.html#why-is-numpy-fast)).
- **Scaling:** Enables calculations over large k-point grids and many bands.
- **Natural indexing:** Output shapes now follow the most common convention (k-points as the leading axis).

**References:**
- [NumPy Best Practices](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [SciPy Lecture Notes: Performance](https://scipy-lectures.org/advanced/optimizing/)

### 6. Output and Display Refactor

**Summary:**  
Display functionality (`display`) was renamed and refactored as `report` for clarity and modularity. Uses `np.array2string` for aligned output.

**Rationale:**
- **Clarity:** Method name reflects its role (reporting summary information).
- **Customizability:** Supports formatting (centering).
- **Consistency:** Aligns with Python dunder methods (`__str__`, `__repr__`).

**References:**
- [PEP 257 – Docstring Conventions](https://peps.python.org/pep-0257/)
- [Python Data Model (dunder methods)](https://docs.python.org/3/reference/datamodel.html#special-method-names)

### 7. Packaging and Distribution

**Summary:**  
Transitioned from `setup.py` to `pyproject.toml`, added standard files (`LICENSE`, `CHANGELOG`, `CONTRIBUTING`).

**Rationale:**
- **PEP 517/518 compliance:** `pyproject.toml` is the new standard for Python packaging.
- **Reproducibility and interoperability:** Tools like Poetry, Flit, and pip rely on standardized metadata.
- **Contributor clarity:** `CONTRIBUTING` and `LICENSE` are now expected in open source.

**References:**
- [Packaging Python Projects – Official Tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [PEP 517 – pyproject.toml build-system](https://peps.python.org/pep-0517/)
- [GitHub Community Standards](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/)

### 8. Python Version Support

**Summary:**  
Dropped support for Python <3.10. Now require Python 3.10+.

**Rationale:**
- **Simplifies code:** Allows use of latest syntax and features (match-case, improved typing, etc.).
- **Upstream dependencies:** Many scientific libraries have dropped older versions.
- **Follows [SPEC-0](https://scientific-python.org/specs/spec-0000/):** Scientific Python minimum standards.

**References:**
- [SPEC-0000](https://scientific-python.org/specs/spec-0000/)
- [Python Release Calendar](https://devguide.python.org/versions/)


### 9. References and Further Reading

- [The Zen of Python](https://peps.python.org/pep-0020/)
- [Effective Python, 2nd Ed. (Brett Slatkin)](https://effectivepython.com/)
- [Refactoring (Martin Fowler)](https://martinfowler.com/books/refactoring.html)
- [Python in a Nutshell (Martelli et al.)](https://www.oreilly.com/library/view/python-in-a/0596100469/)

**If you have questions or want to propose a change, please open an issue or start a discussion.**  
For contributing guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).