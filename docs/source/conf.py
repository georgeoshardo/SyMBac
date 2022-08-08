# Configuration file for the Sphinx documentation builder.

# -- Project information
import os
import sys
sys.path.insert(0, os.path.abspath('../../SyMBac/'))


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
    'sphinx.ext.autosectionlabel'
]

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
