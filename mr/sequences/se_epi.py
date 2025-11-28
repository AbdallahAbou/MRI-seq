"""
Spin Echo EPI Sequence

A single-shot EPI sequence with spin echo refocusing.
Uses zigzag (Cartesian) k-space trajectory.

Reference: pulseq writeEpiSpinEcho.m
"""

import numpy as np
from ..primitives import (
    SystemLimits, Trapezoid,
    make_sinc_pulse, make_block_pulse, make_trapezoid, make_adc, make_delay,
    calc_duration, calc_rf_center
)
from ..sequence import Sequence


def create_se_epi(
    fov: float = 256e-3,
    nx: int = 64,
    ny: int = 64,
    thickness: float = 3e-3,
    te: float = 60e-3,
    system: SystemLimits = None
) -> Sequence:
    """
    Create a Spin Echo EPI sequence
    
    This implements a single-shot EPI with spin echo refocusing.
    The k-space trajectory is a zigzag pattern (Cartesian sampling).
    
    Args:
        fov: Field of view (m), default 256mm
        nx: Number of readout samples, default 64
        ny: Number of phase encode lines, default 64
        thickness: Slice thickness (m), default 3mm
        te: Echo time (s), default 60ms
        system: System limits (uses defaults if None)
    
    Returns:
        Sequence object with SE-EPI blocks
    
    Example:
        from mr.sequences import create_se_epi
        seq = create_se_epi(fov=220e-3, nx=128, ny=128, te=80e-3)
        print(seq)  # Sequence(263 blocks, 165.23 ms)
    """
    if system is None:
        system = SystemLimits()
    
    seq = Sequence(system)
    
    # === RF Pulses ===
    # 90° excitation with slice selection
    rf, gz, _ = make_sinc_pulse(
        flip_angle=np.pi/2,
        duration=3e-3,
        slice_thickness=thickness,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        use='excitation'
    )
    
    # 180° refocusing pulse
    rf180 = make_block_pulse(np.pi, duration=500e-6, system=system, use='refocusing')
    
    # === Readout Gradient and ADC ===
    delta_k = 1 / fov
    k_width = nx * delta_k
    readout_time = 3.2e-4
    
    gx = make_trapezoid('x', flat_area=k_width, flat_time=readout_time, system=system)
    adc = make_adc(nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
    
    # === Prephasing Gradients ===
    pre_time = 8e-4
    
    # Move to -kx_max (negative area)
    gx_pre = make_trapezoid('x', area=-gx.area/2 - delta_k/2, duration=pre_time, system=system)
    # Move to -ky_max (negative area) 
    gy_pre = make_trapezoid('y', area=-ny/2 * delta_k, duration=pre_time, system=system)
    # Slice rephasing
    gz_reph = make_trapezoid('z', area=-gz.area/2, duration=pre_time, system=system)
    
    # === Phase Encode Blip ===
    gy_blip = make_trapezoid('y', area=delta_k, system=system)
    
    # === Spoiler Gradients ===
    # Use longer duration to stay within gradient limits
    gz_spoil = make_trapezoid('z', area=gz.area * 2, system=system)
    
    # === TE Timing Calculation ===
    duration_to_center = (nx/2 + 0.5) * calc_duration(gx) + ny/2 * calc_duration(gy_blip)
    rf_center_incl_delay = rf.delay + calc_rf_center(rf)
    rf180_center_incl_delay = rf180.delay + calc_rf_center(rf180)
    
    delay_te1 = te/2 - calc_duration(gz) + rf_center_incl_delay - pre_time - calc_duration(gz_spoil) - rf180_center_incl_delay
    delay_te2 = te/2 - calc_duration(rf180) + rf180_center_incl_delay - calc_duration(gz_spoil) - duration_to_center
    
    # Ensure positive delays
    delay_te1 = max(delay_te1, system.grad_raster_time)
    delay_te2 = max(delay_te2, system.grad_raster_time)
    
    # === Build Sequence ===
    # Excitation
    seq.add_block(rf, gz)
    
    # Prephasing
    seq.add_block(gx_pre, gy_pre, gz_reph)
    
    # Wait for TE/2
    seq.add_block(make_delay(delay_te1))
    
    # Spoiler before 180
    seq.add_block(gz_spoil)
    
    # 180° refocusing
    seq.add_block(rf180)
    
    # Spoiler after 180
    seq.add_block(gz_spoil)
    
    # Wait for TE/2
    seq.add_block(make_delay(delay_te2))
    
    # EPI readout train (zigzag)
    gx_current = gx
    for i in range(ny):
        seq.add_block(gx_current, adc)
        seq.add_block(gy_blip)
        # Reverse gradient for next line
        gx_current = Trapezoid(
            channel='x',
            amplitude=-gx_current.amplitude,
            rise_time=gx_current.rise_time,
            flat_time=gx_current.flat_time,
            fall_time=gx_current.fall_time,
            delay=gx_current.delay,
            area=-gx_current.area,
            flat_area=-gx_current.flat_area
        )
    
    # Set metadata
    seq.set_definition('FOV', [fov, fov, thickness])
    seq.set_definition('Name', 'se_epi')
    seq.set_definition('TE', te)
    seq.set_definition('Nx', nx)
    seq.set_definition('Ny', ny)
    
    return seq
