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
from .pythtb import *

# Use the core module's __all__ to define the package exports
# This ensures 'from pythtb import *' pulls in the same symbols
from . import pythtb as _core
__all__ = getattr(_core, "__all__", [name for name in globals() if not name.startswith("_ ")])

# if pythtb.py defines __all__, that will control what “import *” pulls in.
# otherwise, Python will export all names that don’t start with an underscore.