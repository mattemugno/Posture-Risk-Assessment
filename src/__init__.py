"""Shared setup for the local Flask application."""

import matplotlib

# Reports render in Flask threads; GUI backends require the main thread on macOS.
matplotlib.use('Agg')
