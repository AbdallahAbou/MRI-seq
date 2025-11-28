"""
Pre-built MRI Sequence Library

This module provides ready-to-use sequence implementations.
Each function returns a Sequence object that can be animated or exported.

Available sequences:
    - se_epi: Spin Echo EPI (zigzag Cartesian trajectory)
    - spiral: Spiral readout (expanding from center)
"""

from .se_epi import create_se_epi
from .spiral import create_spiral

__all__ = [
    'create_se_epi',
    'create_spiral',
]
