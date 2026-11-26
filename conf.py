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
copyright = 'Copyright © 2026, Flavia Giammarino.'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import sys
from pathlib import Path

sys.path.append(str(Path('extensions').resolve()))

extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    "sphinx.ext.todo",
    "sphinx.ext.ifconfig",
    'sphinx.ext.extlinks',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    "sphinx_automodapi.automodapi",
    'sphinx_sitemap',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_parser',
    'ablog'
]

templates_path = ['templates']

exclude_patterns = ['docs', 'Thumbs.db', '.DS_Store']

master_doc = "index"
highlight_language = 'python3'
sitemap_url_scheme = "{link}"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_baseurl = 'https://flaviagiammarino.com/'
html_title = 'FlaviaGiammarino.com | AI/ML Consultant | PhD, AWS-Certified'
html_short_title = 'FlaviaGiammarino.com'
html_favicon = 'static/favicon.png'
html_static_path = ['static']
html_extra_path = ['extra']
html_css_files = ['custom.css']
html_js_files = ['custom.js']
html_show_sourcelink = True
html_use_index = True
html_domain_indices = False

# -- ABlog settings ---------------------------------------------------

blog_title = "blog"

blog_baseurl = f"{html_baseurl}/blog"

blog_authors = {
    "Flavia": ("Flavia Giammarino", "https://flaviagiammarino.com"),
}
blog_default_author = "Flavia"

blog_languages = {
    'en': ('English', None),
}
blog_default_language = "en"

skip_injecting_base_ablog_templates = True

# -- Theme configuration options ---------------------------------------------

html_theme = "shibuya"

html_theme_options = {
    "globaltoc_expand_depth": 1,
    "github_url": "https://github.com/flaviagiammarino",
    "linkedin_url": "https://www.linkedin.com/in/flaviagiammarino"
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "ablog/categories.html",
        "ablog/tagcloud.html",
        "ablog/archives.html",
    ]
}

# -- MyST settings ---------------------------------------------------
myst_heading_anchors = 6

myst_enable_extensions = [
    'dollarmath',
    'colon_fence'
]

# -- MathJax settings ---------------------------------------------------

mathjax3_config = {
    "chtml": {
        "matchFontHeight": False
    },
}
# -- Right sidebar table of contents ---------------------------------------------------
# Shibuya shortens dotted API names in the table of contents by deleting every
# `<span class="pre">word.` it finds, which also truncates section headings
# formatted as code (``model.py`` becomes "py") and leaves the HTML unbalanced.
# Keep the original table of contents and redo Shibuya's processing (dropping
# the page title level) without the deletion.
import xml.etree.ElementTree as ET


def _stash_toc(app, pagename, templatename, context, doctree):
    context["_original_toc"] = context.get("toc")


def _restore_toc(app, pagename, templatename, context, doctree):
    toc = context.pop("_original_toc", None)
    if not toc:
        return
    root = ET.fromstring(toc)
    if len(root) == 1 and len(root[0]) == 2 and root[0][1].tag == "ul":
        toc = ET.tostring(root[0][1], encoding="unicode", method="html")
    context["toc"] = toc


def setup(app):
    # Shibuya's handler runs at the default priority (500)
    app.connect("html-page-context", _stash_toc, priority=100)
    app.connect("html-page-context", _restore_toc, priority=900)
