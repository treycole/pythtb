# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import pythtb
import logging
import warnings
# warnings.filterwarnings("error", category=SyntaxWarning)
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
    'undoc-members': False,
    'no-show-inheritance': True,
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
    # 'myst_parser',        # <-- enable Markdown
    "myst_nb",
    'sphinx.ext.doctest',
   # 'sphinx.ext.imgmath',
    'matplotlib.sphinxext.plot_directive',
    # 'sphinx_thebe',
    'sphinx.ext.mathjax',
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinxcontrib.programoutput"
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
    "attrs_inline"
]

nb_execution_mode = "auto"     # run notebooks when building
nb_execution_timeout = 600     # seconds per notebook

thebe_config = {
    "repository_url": "https://github.com/youruser/yourrepo",
    "repository_branch": "v2",
    # CSS selector for code cells
    "selector": "div.nbinput",
}

copybutton_only_copy_prompt_lines = False
copybutton_remove_prompts = True

# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3', None),
#     'numpy': ('https://numpy.org/doc/stable/', None)
# }

# tell Sphinx to treat .md files as sources
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

# for matplotlib plots
plot_formats=[('png',140),('pdf',140)]
pygments_style = "sphinx"
pygments_dark_style = "monokai"  # for dark theme compatibility

# for autodoc to work on PythTB package
sys.path.append("../src")

html_theme = 'pydata_sphinx_theme' #'sphinx_book_theme' #'classic' pydata_sphinx_theme
html_title = f"{project} Docs"

templates_path = ['_templates']
html_static_path = ['_static']
html_js_files = [
    ("custom-icons.js", {"defer": "defer"}),
    "https://unpkg.com/thebe@latest/lib/index.js",
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]
html_extra_path = ['misc', 'simple_fig']
html_css_files = ["custom.css"]
exclude_patterns = ['generated/*.md', 'examples_rst/*', 'examples_py/*']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_context = {
    "github_user": "treycole",
    "github_repo": "pythtb",
    "github_version": "main",
    "doc_path": "docs",
}

html_theme_options = {
#     "navigation_depth": 4,
#     "collapse_navigation": False,
#     "show_nav_level": 2,
    # "logo": {
    #     "image_light": "_static/logo.svg",
    #     "image_dark": "_static/logo_dark.svg",
    # },
    # "github_url": "https://github.com/treycole/pythtb",
    "collapse_navigation": False,
    # "external_links": [
    #     {"Changelog": "", "url": ""},
    # ],
    "header_links_before_dropdown": 6,
    "show_toc_level": 3,
    # Add light/dark mode and documentation version switcher:
    # "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": [
        "search-button",
        "theme-switcher",
        "navbar-icon-links"
    ],
    "navbar_persistent": [],
    # "switcher": {
    #     "version_match": version,
    #     "json_url": "_static/switcher.json",
    # },
    "show_version_warning_banner": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/treycole/pythtb",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pythtb/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    # "use_thebe": True
}

html_sidebars = {
    "index": [],
    "install": [],
    "about": [],
    "CHANGELOG": [],
    "formalism": [],
}

# html_theme_options["use_thebe"] = True  # e/nables Thebe for notebook
# html_js_files = [
    # "https://unpkg.com/thebe@latest/lib/index.js"
# ] # for executing code

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

