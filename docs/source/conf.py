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
version = pythtb.__version__

# preamble for latex formulas
# pngmath_latex_preamble = r"\usepackage{cmbright}"
# pngmath_dvipng_args = ['-gamma 1.5', '-D 110']
# pngmath_use_preview = True

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_typehints = "description"

# link to numpy and python
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

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
    # "numpydoc"
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

# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3', None),
#     'numpy': ('https://numpy.org/doc/stable/', None)
# }

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

html_theme = 'pydata_sphinx_theme' #'sphinx_book_theme' #'classic'

# html_theme_options = {
#     "show_toc_level": 2,
#     "navigation_depth": 4,
#     "collapse_navigation": False,
#     "show_nav_level": 2,
#     # Optional: sticky nav
#     "navbar_start": ["navbar-logo", "version-switcher"],
# }

html_theme_options = {
    # "logo": {
    #     "image_light": "_static/numpylogo.svg",
    #     "image_dark": "_static/numpylogo_dark.svg",
    # },
    "github_url": "https://github.com/sinisacoh/pythtb",
    "collapse_navigation": False,
    # "external_links": [
    #     {"name": "Learn", "url": "https://numpy.org/numpy-tutorials/"},
    #     {"name": "NEPs", "url": "https://numpy.org/neps"},
    # ],
    "header_links_before_dropdown": 6,
    "show_toc_level": 3,
    # Add light/dark mode and documentation version switcher:
    "navbar_end": [
        "search-button",
        "theme-switcher",
        "version-switcher",
        "navbar-icon-links"
    ],
    "navbar_persistent": [],
    "switcher": {
        "version_match": version,
        "json_url": "versions.json",
    },
    "show_version_warning_banner": True,
}

html_static_path = ['_static']
# html_theme_options["use_thebe"] = True  # e/nables Thebe for notebook
# html_js_files = [
    # "https://unpkg.com/thebe@latest/lib/index.js"
# ] # for executing code
html_title = f"{project} v{version} Docs"

# documentation.
# html_theme_options = {
#     # "repository_url": "https://github.com/sinisacoh/pythtb",
#     # "use_repository_button": True,
#     # "use_issues_button": True,
#     # "use_edit_page_button": True,
#     # "path_to_docs": "docs/source",
#     # "repository_branch": "main",
#     # "use_download_button": True,
#     # "home_page_in_toc": True,
#     "show_toc_level": 2,
# }
# html_sidebars = {
#           '**':    ['globaltoc.html', 'localtoc.html', 'searchbox.html'],
#           'index': ['globaltoc.html', 'searchbox.html'],
#        }
# remove "show source" from website
html_copy_source=False
html_show_sourcelink=False

# The master toctree document.
master_doc = 'index'

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

