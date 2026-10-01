"""Hummel backend for law.

The backend presents autonomous humsub chains as stable remote jobs to law.
"""

from .job import HummelJobManager
from .workflow import HummelWorkflow

__all__ = ["HummelJobManager", "HummelWorkflow"]
