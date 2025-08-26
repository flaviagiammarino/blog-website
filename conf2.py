# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = ''
author = 'Flavia Giammarino'
release = '2025-08-25'
language = 'en'
copyright = 'Copyright © 2025, Flavia Giammarino'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import sys
from pathlib import Path

sys.path.append(str(Path('extensions').resolve()))

extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.viewcode',
    'sphinx_sitemap',
    'myst_parser',
    'sphinx.ext.mathjax',
    'ablog',
    'extensions'
]

templates_path = []

exclude_patterns = ['Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_baseurl = 'https://flaviagiammarino.com/'
html_title = 'flaviagiammarino.com'
html_short_title = 'flaviagiammarino.com'
html_logo = 'static/logo.png'
html_favicon = 'static/favicon.ico'
html_static_path = ['static']
html_extra_path = ['extra']
html_css_files = ['custom.css']
html_js_files = ['custom.js']
html_show_sourcelink = False
highlight_language = 'python3'
sitemap_url_scheme = "{link}"
myst_heading_anchors = 6

# -- Theme configuration options ---------------------------------------------
# html_theme = 'shibuya'

html_theme_options = {
    "globaltoc_expand_depth": 10,
}

# -- ABlog settings ---------------------------------------------------
html_sidebars = {
   "**": [
      # Ablog sidebars
      "ablog/postcard.html",
      "ablog/recentposts.html",
      "ablog/tagcloud.html",
      "ablog/categories.html",
      "ablog/archives.html",
      "ablog/authors.html",
      "ablog/languages.html",
      "ablog/locations.html",
   ]
}

# -- MyST settings ---------------------------------------------------
myst_enable_extensions = [
    'dollarmath',
    'colon_fence'
]
