# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys, os
import pythtb
import logging
logging.getLogger("pythtb").setLevel(logging.WARNING)

project = u'PythTB'
copyright = '2025, Trey Cole, Sinisa Coh, and David Vanderbilt'
author = 'Trey Cole, Sinisa Coh, and David Vanderbilt'
release = pythtb.__version__

# preamble for latex formulas
# pngmath_latex_preamble = r"\usepackage{cmbright}"
# pngmath_dvipng_args = ['-gamma 1.5', '-D 110']
# pngmath_use_preview = True

# autosummary_generate = True
# autodoc_typehints = "description"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  
    'sphinx.ext.viewcode',
    "sphinx.ext.autosummary",
    'myst_parser',        # <-- enable Markdown
    'sphinx.ext.doctest',
   # 'sphinx.ext.imgmath',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.mathjax',
    "sphinx.ext.intersphinx",
]

myst_enable_extensions = [
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "dollarmath",
    "amsmath",
    "substitution",
    "colon_fence",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# tell Sphinx to treat .md files as sources
source_suffix = {
    '.md': 'markdown',
    '.rst': 'restructuredtext',
}

# for matplotlib plots
plot_formats=[('png',140),('pdf',140)]

# for autodoc to work on PythTB package
sys.path.append("../src")

# (optional) tweak MyST syntax

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
# html_theme = 'classic'
html_static_path = ['_static']
# documentation.
html_theme_options = {
    # "repository_url": "https://github.com/sinisacoh/pythtb",
    # "use_repository_button": True,
    # "use_issues_button": True,
    # "use_edit_page_button": True,
    # "path_to_docs": "docs/source",
    # "repository_branch": "main",
    # "use_download_button": True,
    # "home_page_in_toc": True,
    "show_toc_level": 2,
}
# html_sidebars = {
#           '**':    ['globaltoc.html', 'localtoc.html', 'searchbox.html'],
#           'index': ['globaltoc.html', 'searchbox.html'],
#        }
# remove "show source" from website
html_copy_source=False
html_show_sourcelink=False

# The master toctree document.
master_doc = 'index'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# The shorter version.
version = '2.0.0'
# The full version, including alpha/beta/rc tags.
release = '2.0.0'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# preamble for latex formulas
pngmath_latex_preamble=r"\usepackage{cmbright}"
pngmath_dvipng_args=['-gamma 1.5', '-D 110']
pngmath_use_preview=True

# Output file base name for HTML help builder.
htmlhelp_basename = 'PythTBdoc'

# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'PythTB.tex', u'PythTB Documentation',
   u'Sinisa Coh and David Vanderbilt', 'manual'),
]

man_pages = [
    ('index', 'pythtb', u'PythTB Documentation',
     [u'Sinisa Coh and David Vanderbilt'], 1)
]

texinfo_documents = [
  ('index', 'PythTB', u'PythTB Documentation',
   u'Sinisa Coh and David Vanderbilt', 'PythTB', 'Python software package implementation of tight-binding approximation',
   'Miscellaneous'),
]

# for autodoc so that things are ordered as in source
autodoc_member_order = 'bysource' 


# In order to skip some functions in documentation
def maybe_skip_member(app, what, name, obj, skip, options):
    if name in ["tbmodel","add_hop","set_sites","no_2pi"]:
        return True
    else:
        return skip
def setup(app):
    app.connect('autodoc-skip-member', maybe_skip_member)

