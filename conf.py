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
copyright = 'Copyright © 2025, Flavia Giammarino.'

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
    "sphinx_automodapi.automodapi",
    'sphinx_sitemap',
    'myst_parser',
    'sphinx.ext.mathjax',
    'ablog',
    'extensions'
]

templates_path = ['templates']

exclude_patterns = ['docs', 'Thumbs.db', '.DS_Store']

master_doc = "index"
highlight_language = 'python3'
sitemap_url_scheme = "{link}"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_baseurl = 'https://flaviagiammarino.com/'
html_title = 'flaviagiammarino.com'
html_short_title = 'flaviagiammarino.com'
html_favicon = 'static/favicon.ico'
html_static_path = ['static']
html_extra_path = ['extra']
html_css_files = ['custom.css']
html_js_files = ['custom.js']
html_show_sourcelink = True
html_use_index = True
html_domain_indices = False

# -- ABlog settings ---------------------------------------------------

blog_title = "Blog"
blog_baseurl = f"{html_baseurl}/blog"
blog_authors = {
    "Flavia": ("Flavia Giammarino", "https://flaviagiammarino.com"),
}
blog_default_author = "Flavia"
blog_feed_archives = True
blog_feed_fulltext = True
blog_feed_templates = {
    "atom": {
        "content": "{{ title }}{% for tag in post.tags %} #{{ tag.name|trim()|replace(' ', '') }}{% endfor %}",
    },
    "social": {
        "content": "{{ title }}{% for tag in post.tags %} #{{ tag.name|trim()|replace(' ', '') }}{% endfor %}",
    },
}
skip_injecting_base_ablog_templates = True

# -- Theme configuration options ---------------------------------------------

html_theme = "shibuya"

html_theme_options = {
    "globaltoc_expand_depth": 1,
    "light_logo": "_static/icon-light.png",
    "dark_logo": "_static/icon-dark.png",
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "ablog/categories.html",
        "ablog/tagcloud.html",
        "ablog/archives.html",
        "ablog/authors.html",
    ]
}

# -- MyST settings ---------------------------------------------------
myst_heading_anchors = 6

myst_enable_extensions = [
    'dollarmath',
    'colon_fence'
]
