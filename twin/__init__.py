"""Drift Containment & Remediation platform — the twin package.

Public surface: `Engine` (facade) plus the models. See the module docstrings for
how each piece maps to the builder brief sections.
"""
from .engine import Engine

__all__ = ["Engine"]
__version__ = "0.1.0"
