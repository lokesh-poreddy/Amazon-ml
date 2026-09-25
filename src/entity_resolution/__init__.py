"""
Amazon ML Challenge 2026 — Business Entity Resolution Engine.

Architecture:
    NORMALIZE → RETRIEVE → SCORE → DECIDE → SUBMIT

Every component is pluggable. Advanced methods extend interfaces,
they do not replace the pipeline.
"""

__version__ = "1.0.0"
