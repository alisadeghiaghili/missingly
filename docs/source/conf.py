# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# ``nbsphinx`` invokes Pandoc as a subprocess.  The docs extra ships a
# platform-specific binary through pypandoc-binary, whose directory is not
# automatically added to PATH in every environment (notably Windows CI).
try:
    import pypandoc

    _pandoc_dir = os.path.dirname(pypandoc.get_pandoc_path())
    os.environ['PATH'] = _pandoc_dir + os.pathsep + os.environ.get('PATH', '')
except (ImportError, OSError):
    # nbsphinx will emit the actionable PandocMissing error if the docs extra
    # was not installed; keeping config importable helps partial API builds.
    pass

# -- Project information -----------------------------------------------------

project = 'missingly'
copyright = '2026, Ali Sadeghi Aghili'
author = 'Ali Sadeghi Aghili'

version = '1.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'nbsphinx',
]

# Napoleon settings (for Google-style docstrings too)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# numpydoc settings
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'description'
autodoc_mock_imports = [
    'plotly', 'upsetplot', 'arabic_reshaper', 'bidi',
    'data_quality_toolkit',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_title = 'missingly 1.0.0'
html_short_title = 'missingly'

html_theme_options = {
    'source_repository': 'https://github.com/alisadeghiaghili/missingly',
    'source_branch': 'main',
    'source_directory': 'docs/source/',
}

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}
