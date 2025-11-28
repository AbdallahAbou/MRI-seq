"""
MR Pulse Sequence Library
========================

A modular Python library for creating MRI pulse sequences.
Direct port of core pulseq MATLAB functionality.

Modules:
    primitives: Core building blocks (RF pulses, gradients, ADC)
    sequence: Sequence container and manipulation
    sequences: Pre-built sequence implementations
    animator: Visualization and animation tools

Usage:
    from mr import SystemLimits, Sequence
    from mr import make_sinc_pulse, make_trapezoid, make_adc
    from mr.sequences import create_se_epi, create_spiral
    from mr.animator import SequenceAnimator, animate_sequence
"""

from .primitives import (
    SystemLimits,
    RFPulse,
    Trapezoid,
    ArbitraryGradient,
    ADC,
    Delay,
    make_sinc_pulse,
    make_block_pulse,
    make_trapezoid,
    make_arbitrary_grad,
    make_adc,
    make_delay,
    calc_duration,
    calc_rf_center,
)

from .sequence import Sequence
from .animator import SequenceAnimator, animate_sequence, plot_sequence

__all__ = [
    # Classes
    'SystemLimits',
    'RFPulse', 
    'Trapezoid',
    'ArbitraryGradient',
    'ADC',
    'Delay',
    'Sequence',
    'SequenceAnimator',
    # Functions
    'make_sinc_pulse',
    'make_block_pulse',
    'make_trapezoid',
    'make_arbitrary_grad',
    'make_adc',
    'make_delay',
    'calc_duration',
    'calc_rf_center',
    'animate_sequence',
    'plot_sequence',
]
