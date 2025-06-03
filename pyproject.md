[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pythtb"
version = "2.0.0"
description = "Solver for tight binding models in condensed matter physics and materials science."
readme = "README.md"  
requires-python = ">=3.11"
authors = [
  { name = "Trey Cole", email = "trey@treycole.me" },
  { name = "Sinisa Coh", email = "sinisacoh@gmail.com" },
  { name = "David Vanderbilt", email = "dhv@physics.rutgers.edu" }
]
license = { file = "LICENSE" }  # or use: license = { text = "GPL-3.0" }
keywords = ["tight binding", "solid state physics", "condensed matter physics", "materials science"]
dependencies = [
  "numpy",
  "matplotlib"
]
classifiers = [
  "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
  "Programming Language :: Python :: 3",
  "Operating System :: OS Independent"
]
urls = { 
  "Homepage" = "http://www.physics.rutgers.edu/pythtb",
  "Download" = "http://www.physics.rutgers.edu/pythtb"
}

[tool.setuptools.packages.find]
where = ["."]
include = ["pythtb*"]
