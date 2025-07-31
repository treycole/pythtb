
# File structure

- `pythtb/`: PythTB package source
  - tb_model.py - contains `TBModel` class
  - wf_array.py - contains `WFArray` class
  - bloch.py - contains `Bloch` class
  - wannier.py - contains `Wannier` class
  - k_mesh.py - contains `KMesh` class
  - utils.py - utility functions used throughout
  - visualization.py - visualization functions 
- `examples/`: folder with examples
- `dev/talk/`: presentation on Python and PythTB
- `dev/formalism/`: LaTeX document describing PythTB formalism
- `website/`: source of website for PythTB written in Sphinx.
    - To create website from scratch and generate source tar files, run script "go"
    - To open website in browser run script "see"
    - For quick update to website run "make html"
    - To clean up website and all other files run "clean"
    - To copy files to physsun run "publish"

Do NOT edit following folders directly:
- website/source/examples
- website/source/misc
