# Contributing to `pythTB`

Thank you for your interest in contributing to `pythTB`! Your contributions help us improve the project for all users. Please review the following guidelines before submitting your changes.

## Code Quality and Design

- **Clarity and Maintainability:**  
  Write code that is straightforward and easy to understand. Our aim is to produce reliable and maintainable code—simplicity in design is highly valued.

- **Comprehensive Documentation:**  
  Every edge-case should be clearly documented. Whether it’s a non-obvious behavior, non-standard convention, or any special case, please explain it in the code comments and documentation.

- **Simple Class Structure:**  
  When designing new classes or refactoring existing ones, strive for a minimal and intuitive interface. Avoid unnecessary complexity; the simpler the design, the easier it is to maintain and extend.

- **Handling Ambiguity:**  
  In situations where a function could produce ambiguous output, it is better to print a warning and refrain from returning a potentially confusing result. This approach helps maintain trust and clarity for the end user.

- **Vectorizing:**
   Where possible, avoid nested `for` loops in favor of vectorized operations, especially for linear algebra operations. This helps avoid bottlenecks and keeps the code running smoothly and effeciently.  

## How to Contribute

1. **Fork the Repository:**  
   Create your own fork of the repository on GitHub.

2. **Create a Feature or Bugfix Branch:**  
   Use a descriptive branch name for your changes (e.g., `feature/add-new-solver` or `bugfix/fix-edge-case-handling`).

3. **Write and Update Tests:**  
   Add tests for any new functionality and ensure existing tests pass. Cover edge-cases diligently.

4. **Document Your Changes:**  
   Update the documentation, comments, and any relevant user guides to reflect your changes. Clear documentation is essential for future maintenance.

5. **Submit a Pull Request:**  
   When your changes are ready, submit a pull request. Include a clear description of what you’ve done and why. This helps maintainers review and merge your changes effectively.

## Reporting Issues

If you encounter bugs or have suggestions for improvements:
- Please open an issue on GitHub.
- Provide as much detail as possible (e.g., minimal example, steps to reproduce, error messages, and relevant system information).

## Code Reviews

All pull requests will undergo a code review process. Feedback may include suggestions for further improvements or clarifications.

## Final Note

Our goal is to make **pythtb** as clear and robust as possible. By focusing on well-documented, maintainable code and handling ambiguous cases with caution, we ensure that the project remains reliable and easy to use for everyone.

Thank you for contributing and helping us improve **pythtb**!
