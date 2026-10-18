# Configuration file for Sphinx documentation (Python API).
# Run from the directory that contains the staged ``libspeech`` package (the build tree root):
# docs/python/CreateDocs.cmake does that.

from __future__ import annotations

import contextlib
import re
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent.parent.parent
LOGO = PROJECT_ROOT / "docs" / "logo" / "libspeech-logo.svg"

# ── Version (include/libspeech/version.h is the single source of truth) ──────
version = "0.0.0"
_header = PROJECT_ROOT / "include" / "libspeech" / "version.h"
if _header.exists():
    _text = _header.read_text()
    _parts = [re.search(rf"#define\s+LIBSPEECH_VERSION_{n}\s+(\d+)", _text) for n in ("MAJOR", "MINOR", "PATCH")]
    if all(_parts):
        version = ".".join(m.group(1) for m in _parts)

sys.path.insert(0, ".")
import libspeech  # noqa: E402

project = "libspeech Python"
copyright = "2025, Mohammad Raziei"
author = "Mohammad Raziei"
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]
autodoc_member_order = "groupwise"
autodoc_typehints = "description"
autosummary_generate = True
add_module_names = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# The compiled modules are an implementation detail (libspeech.speech_dsp_py, ...): show every
# public name as ``libspeech.<name>``.
_PRIVATE = ("speech_dsp_py", "speech_io_py", "speech_models_py", "_about")


def _rename(text):
    for mod in _PRIVATE:
        text = text.replace(f"libspeech.{mod}.", "libspeech.")
    return text


def _fix_docstring(app, what, name, obj, options, lines):
    lines[:] = [_rename(line) for line in lines]


def _fix_signature(app, what, name, obj, options, signature, return_annotation):
    return (_rename(signature) if signature else signature,
            _rename(return_annotation) if return_annotation else return_annotation)


def setup(app):
    app.connect("autodoc-process-docstring", _fix_docstring)
    app.connect("autodoc-process-signature", _fix_signature)
    for mod_name in _PRIVATE:
        mod = getattr(libspeech, mod_name, None)
        if mod is None:
            continue
        for attr in dir(mod):
            obj = getattr(mod, attr, None)
            if getattr(obj, "__module__", None) == f"libspeech.{mod_name}":
                with contextlib.suppress(AttributeError, TypeError):
                    obj.__module__ = "libspeech"


language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "speech"
html_theme_path = [str(DOCS_DIR / "_themes")]
html_css_files = []
pygments_style = "monokai"
if LOGO.exists():
    html_logo = str(LOGO)
html_last_updated_fmt = "%Y-%m-%d %H:%M"
