"""Production WSGI entry point.

Run with:  gunicorn wsgi:server

Puts ``gui/`` on the import path (``ce_viz`` adds ``src/`` itself) and exposes
the Dash app's underlying Flask server as ``server``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gui"))

from app import server  # noqa: E402,F401  (imported for gunicorn)
