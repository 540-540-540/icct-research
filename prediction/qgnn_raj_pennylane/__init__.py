"""PennyLane-native Raj QGNN P1/P1.1 public surface.

Residual/model.py and classical.py are retained as unapproved P2 prototypes but
are intentionally not exported until the P2 interface is re-frozen.
"""
from .quantum import PennyLaneRajBranch, PennyLaneRajMultiJCore

__all__ = ["PennyLaneRajBranch", "PennyLaneRajMultiJCore"]
