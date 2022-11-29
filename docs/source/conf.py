# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys
from sphinx_automodapi import automodsumm
from sphinx_automodapi.utils import find_mod_objs
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../examples'))


project = 'SyMBac'
copyright = '2022, Georgeos Hardo'
author = 'Georgeos Hardo'

release = '0.2'
version = '0.2.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx.ext.mathjax"
]

autoapi_dirs = ['../../SyMBac']
nbsphinx_execute = 'never'


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Options for plot output
# -----------------------
plot_include_source = True

# numpydoc stuff
numpydoc_class_members_toctree = True
numpydoc_xref_param_type = True
numpydoc_validation_checks = {"all", "GL01", "SA04", "RT03"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "source/examples/data/*", "source/examples/data/masks/", "source/examples/data/scenes/"]
pygments_style = "sphinx"
