"""Provides assets for forging analysis datasets from processed experimental data."""

from ataraxis_base_utilities import console

# Ensures console and progress bars are enabled when this library is used.
if not console.enabled:
    console.enable()
if not console.progress_enabled:
    console.enable_progress()
