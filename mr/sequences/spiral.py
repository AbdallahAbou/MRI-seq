"""
Spiral Readout Sequence

A gradient echo sequence with spiral k-space trajectory.
Implements Archimedean spiral starting from k-space center and spiraling outward.

Reference: https://pulseq.github.io/writeSpiral.html
"""

import numpy as np
from ..primitives import (
    SystemLimits, ArbitraryGradient,
    make_sinc_pulse, make_trapezoid, make_adc, make_delay, make_arbitrary_grad,
    calc_duration
)
from ..sequence import Sequence


def traj2grad(k: np.ndarray, raster_time: float = 10e-6) -> tuple:
    """
    Convert k-space trajectory to gradient waveforms
    
    Port of mr.traj2grad from pulseq MATLAB
    
    Args:
        k: K-space trajectory array (2, N) with kx and ky
        raster_time: Gradient raster time (s)
    
    Returns:
        Tuple of (gradient, slew_rate) arrays
    """
    # Gradient is derivative of k-space
    # g = dk/dt 
    
    # Calculate gradient from k-space derivative
    dk = np.diff(k, axis=1)
    g = dk / raster_time
    
    # Prepend zero to match length
    g = np.hstack([np.zeros((2, 1)), g])
    
    # Calculate slew rate
    dg = np.diff(g, axis=1)
    s = dg / raster_time
    s = np.hstack([np.zeros((2, 1)), s])
    
    return g, s


def design_archimedean_spiral(
    fov: float,
    nx: int,
    oversampling: int = 2,
    system: SystemLimits = None
) -> tuple:
    """
    Design an Archimedean spiral trajectory (pulseq style)
    
    Based on writeSpiral.m from pulseq
    
    Args:
        fov: Field of view (m)
        nx: Matrix size
        oversampling: Oversampling factor (default 2)
        system: System limits
    
    Returns:
        Tuple of (gx, gy, adc_samples, trajectory_kx, trajectory_ky)
    """
    if system is None:
        system = SystemLimits()
    
    # K-space parameters
    delta_k = 1 / fov
    k_radius = nx // 2
    k_samples = round(2 * np.pi * k_radius) * oversampling
    
    # Calculate raw Archimedean spiral trajectory
    # k(c) = r * exp(i*a) where r = delta_k * c / k_samples
    total_points = k_radius * k_samples + 1
    ka = np.zeros((2, total_points))
    
    for c in range(total_points):
        r = delta_k * c / k_samples
        a = (c % k_samples) * 2 * np.pi / k_samples
        ka[0, c] = r * np.cos(a)  # kx
        ka[1, c] = r * np.sin(a)  # ky
    
    # Calculate gradients and slew rates from trajectory
    ga, sa = traj2grad(ka, system.grad_raster_time)
    
    # Apply safety margin for slew rate limits
    safety_margin = 0.94
    
    # Calculate time stepping based on gradient and slew rate limits
    # Magnitude-based limits (smooth constraint)
    g_mag = np.abs(ga[0, :] + 1j * ga[1, :])
    s_mag = np.abs(sa[0, :] + 1j * sa[1, :])
    
    dt_g_abs = g_mag / (system.max_grad * safety_margin) * system.grad_raster_time
    dt_s_abs = np.sqrt(s_mag / (system.max_slew * safety_margin + 1e-10)) * system.grad_raster_time
    
    # Combined time stepping
    dt_smooth = np.maximum(dt_g_abs, dt_s_abs)
    
    # Apply minimum time step to preserve trajectory detail
    dt_min = 4 * system.grad_raster_time / k_samples
    dt_smooth = np.maximum(dt_smooth, dt_min)
    
    # Ensure no zeros or NaN
    dt_smooth = np.maximum(dt_smooth, system.grad_raster_time)
    
    # Calculate cumulative time
    t_smooth = np.concatenate([[0], np.cumsum(dt_smooth)])
    
    # Interpolate trajectory to gradient raster time
    n_points = int(np.floor(t_smooth[-1] / system.grad_raster_time))
    if n_points < 2:
        n_points = ka.shape[1]
        kopt = ka
    else:
        t_new = np.arange(n_points) * system.grad_raster_time
        kopt = np.zeros((2, n_points))
        # t_smooth has length len(ka[0,:]) + 1, we need to match ka length
        t_interp = t_smooth[:-1]  # Remove last point to match ka
        kopt[0, :] = np.interp(t_new, t_interp, ka[0, :])
        kopt[1, :] = np.interp(t_new, t_interp, ka[1, :])
    
    # Calculate final gradients from optimized trajectory
    gos, _ = traj2grad(kopt, system.grad_raster_time)
    
    # Extend by one sample (for ADC tuning delay)
    gos = np.hstack([gos, gos[:, -1:]])
    
    return gos[0, :], gos[1, :], k_radius * k_samples, kopt[0, :], kopt[1, :]


def create_spiral(
    fov: float = 256e-3,
    nx: int = 96,
    thickness: float = 3e-3,
    oversampling: int = 2,
    flip_angle: float = None,
    system: SystemLimits = None
) -> Sequence:
    """
    Create a spiral gradient echo sequence
    
    Implements Archimedean spiral trajectory as per pulseq writeSpiral.m
    The k-space trajectory starts at the center and spirals outward.
    
    Args:
        fov: Field of view (m), default 256mm
        nx: Matrix size, default 96
        thickness: Slice thickness (m), default 3mm
        oversampling: K-space oversampling factor, default 2
        flip_angle: Flip angle (rad), default pi/2 (90°)
        system: System limits (uses defaults if None)
    
    Returns:
        Sequence object with spiral readout
    
    Example:
        from mr.sequences import create_spiral
        seq = create_spiral(fov=256e-3, nx=96)
        seq.plot_kspace_2d()
    """
    if system is None:
        # Use pulseq spiral defaults
        system = SystemLimits(
            max_grad=30e-3 * 42.576e6,   # 30 mT/m
            max_slew=120 * 42.576e6,      # 120 T/m/s
            rf_ringdown_time=30e-6,
            rf_dead_time=100e-6,
            adc_dead_time=10e-6
        )
    
    if flip_angle is None:
        flip_angle = np.pi / 2  # 90 degrees
    
    seq = Sequence(system)
    
    # === RF Excitation ===
    rf, gz, gz_reph = make_sinc_pulse(
        flip_angle=flip_angle,
        duration=3e-3,
        slice_thickness=thickness,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        use='excitation'
    )
    
    # === Design Spiral Trajectory ===
    gx_waveform, gy_waveform, adc_samples, kx_traj, ky_traj = design_archimedean_spiral(
        fov=fov,
        nx=nx,
        oversampling=oversampling,
        system=system
    )
    
    # Create arbitrary gradient objects with delay for slice rephasing
    gz_reph_dur = calc_duration(gz_reph)
    gx_spiral = make_arbitrary_grad('x', gx_waveform, system=system, delay=gz_reph_dur)
    gy_spiral = make_arbitrary_grad('y', gy_waveform, system=system, delay=gz_reph_dur)
    
    # ADC during spiral readout
    adc_time = system.grad_raster_time * len(gx_waveform)
    adc_dwell = adc_time / adc_samples
    # Round to ADC raster (100ns typically)
    adc_raster = 100e-9
    adc_dwell = round(adc_dwell / adc_raster) * adc_raster
    if adc_dwell < adc_raster:
        adc_dwell = adc_raster
    
    adc = make_adc(adc_samples, dwell=adc_dwell, delay=gz_reph_dur, system=system)
    
    # === Spoiler gradients ===
    delta_k = 1 / fov
    gz_spoil = make_trapezoid('z', area=delta_k * nx * 4, system=system)
    
    # === Build Sequence ===
    # Excitation
    seq.add_block(rf, gz)
    
    # Spiral readout with slice rephasing
    seq.add_block(gz_reph, gx_spiral, gy_spiral, adc)
    
    # Spoilers
    seq.add_block(gz_spoil)
    
    # Set metadata
    seq.set_definition('FOV', [fov, fov, thickness])
    seq.set_definition('Name', 'spiral')
    seq.set_definition('Nx', nx)
    seq.set_definition('Oversampling', oversampling)
    
    return seq
