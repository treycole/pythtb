"""
PythTB: Python Tight Binding module: a Python package for tight-binding calculations.

May 22, 2025

Copyright (c) 2010-2025 Trey Cole, Sinisa Coh and David Vanderbilt.

PythTB is free software available at http://www.physics.rutgers.edu/pythtb/ .
You can redistribute it and/or modify it under the terms of the GNU General 
Public License v3.0 or later. See LICENSE or https://www.gnu.org/licenses/gpl-3.0.en.html 
for details. 
"""

__version__ = "2.0.0"
__author__ = "Trey Cole, Sinisa Coh, David Vanderbilt"
__license__ = "GPL-3.0"

# Import all public API from the core module
from .tb_model import *
from .wf_array import *
from .w90 import *
from .k_mesh import *
from .wannier import *
from .utils import *

from . import tb_model, wf_array, w90, k_mesh, wannier, utils

__all__ = []
__all__ += tb_model.__all__
__all__ += wf_array.__all__
__all__ += w90.__all__
__all__ += k_mesh.__all__
__all__ += wannier.__all__
__all__ += utils.__all__

# Use the core module's __all__ to define the package exports from * imports.
# This ensures 'from pythtb import *' pulls in only the intended public API.
# If you want to control what gets imported with "from pythtb import *",
# you can define __all__ in the respective modules (tb_model, wf_array, w90).
# This is a common practice in Python packages to avoid polluting the namespace
# with internal details and to provide a clear public API.