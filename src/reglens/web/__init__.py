"""The web interface: a small local server (``scripts/serve.py``) and a single page that talks to it.

``service`` holds every decision (validation, the comparison with the model alone, error messages) as plain Python;
``app`` is the thin HTTP layer on top of it; ``static`` is the page.
"""
