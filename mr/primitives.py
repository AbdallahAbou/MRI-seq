"""
MR Primitives - Core building blocks for MRI pulse sequences

This module provides the fundamental components for building MRI sequences:
- SystemLimits: Hardware constraints
- RFPulse: RF excitation/refocusing pulses
- Trapezoid: Trapezoidal gradients
- ArbitraryGradient: Arbitrary gradient waveforms
- ADC: Analog-to-digital converter readouts
- Delay: Simple timing delays

Each component has associated "make_*" factory functions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, List


# =============================================================================
# System Configuration
# =============================================================================

@dataclass
class SystemLimits:
    """
    MRI system hardware limits
    
    Attributes:
        max_grad: Maximum gradient amplitude (Hz/m)
        max_slew: Maximum gradient slew rate (Hz/m/s)
        rf_dead_time: Dead time before RF pulse (s)
        rf_ringdown_time: Ringdown time after RF pulse (s)
        adc_dead_time: Dead time before ADC (s)
        grad_raster_time: Gradient raster time (s)
        rf_raster_time: RF raster time (s)
        gamma: Gyromagnetic ratio (Hz/T)
    
    Example:
        # Siemens-like system
        sys = SystemLimits(
            max_grad=32e-3 * 42.576e6,  # 32 mT/m
            max_slew=130 * 42.576e6,     # 130 T/m/s
        )
    """
    max_grad: float = 32e-3 * 42.576e6      # 32 mT/m in Hz/m
    max_slew: float = 130 * 42.576e6        # 130 T/m/s in Hz/m/s  
    rf_dead_time: float = 100e-6            # 100 us
    rf_ringdown_time: float = 20e-6         # 20 us
    adc_dead_time: float = 20e-6            # 20 us
    grad_raster_time: float = 10e-6         # 10 us
    rf_raster_time: float = 1e-6            # 1 us
    gamma: float = 42.576e6                 # Hz/T
    
    @classmethod
    def from_mT(cls, max_grad_mT_m: float = 32, max_slew_T_m_s: float = 130, **kwargs):
        """Create SystemLimits from mT/m and T/m/s units"""
        gamma = kwargs.pop('gamma', 42.576e6)
        return cls(
            max_grad=max_grad_mT_m * 1e-3 * gamma,
            max_slew=max_slew_T_m_s * gamma,
            gamma=gamma,
            **kwargs
        )


# =============================================================================
# Event Dataclasses
# =============================================================================

@dataclass
class RFPulse:
    """
    RF pulse event
    
    Attributes:
        signal: Complex signal amplitude array
        t: Time points array (s)
        delay: Delay before pulse (s)
        freq_offset: Frequency offset (Hz)
        phase_offset: Phase offset (rad)
        dead_time: Dead time (s)
        ringdown_time: Ringdown time (s)
        shape_dur: Duration of the shaped part (s)
        use: 'excitation', 'refocusing', or 'inversion'
        center: Time of RF center/peak (s)
    """
    signal: np.ndarray
    t: np.ndarray
    delay: float = 0
    freq_offset: float = 0
    phase_offset: float = 0
    dead_time: float = 0
    ringdown_time: float = 0
    shape_dur: float = 0
    use: str = 'excitation'
    center: float = 0
    
    @property
    def duration(self) -> float:
        """Total duration including delays"""
        return self.delay + self.shape_dur + self.ringdown_time


@dataclass
class Trapezoid:
    """
    Trapezoidal gradient event
    
    Attributes:
        channel: Gradient axis ('x', 'y', or 'z')
        amplitude: Gradient amplitude (Hz/m)
        rise_time: Ramp up time (s)
        flat_time: Flat top time (s)
        fall_time: Ramp down time (s)
        delay: Delay before gradient (s)
        area: Total gradient area (Hz/m * s)
        flat_area: Area during flat portion (Hz/m * s)
    """
    channel: str
    amplitude: float
    rise_time: float
    flat_time: float
    fall_time: float
    delay: float = 0
    area: float = 0
    flat_area: float = 0
    
    @property
    def duration(self) -> float:
        """Total duration including delay"""
        return self.delay + self.rise_time + self.flat_time + self.fall_time
    
    def get_waveform(self, dt: float = 10e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Get time and amplitude arrays for this gradient"""
        t = np.array([
            0,
            self.rise_time,
            self.rise_time + self.flat_time,
            self.rise_time + self.flat_time + self.fall_time
        ]) + self.delay
        amp = np.array([0, self.amplitude, self.amplitude, 0])
        return t, amp


@dataclass
class ArbitraryGradient:
    """
    Arbitrary gradient waveform
    
    Attributes:
        channel: Gradient axis ('x', 'y', or 'z')
        waveform: Gradient amplitude array (Hz/m)
        t: Time points array (s)
        delay: Delay before gradient (s)
    """
    channel: str
    waveform: np.ndarray
    t: np.ndarray
    delay: float = 0
    
    @property
    def duration(self) -> float:
        """Total duration including delay"""
        return self.delay + self.t[-1] if len(self.t) > 0 else self.delay
    
    @property
    def area(self) -> float:
        """Calculate gradient area using trapezoidal integration"""
        return np.trapz(self.waveform, self.t)


@dataclass  
class ADC:
    """
    ADC readout event
    
    Attributes:
        num_samples: Number of samples to acquire
        dwell: Dwell time per sample (s)
        delay: Delay before first sample (s)
        freq_offset: Frequency offset (Hz)
        phase_offset: Phase offset (rad)
        duration: Total duration (s)
        dead_time: Dead time (s)
    """
    num_samples: int
    dwell: float
    delay: float = 0
    freq_offset: float = 0
    phase_offset: float = 0
    duration: float = 0
    dead_time: float = 0
    
    @property
    def total_duration(self) -> float:
        """Total duration including delay and dead time"""
        return self.delay + self.num_samples * self.dwell + self.dead_time


@dataclass
class Delay:
    """Simple delay event"""
    delay: float
    
    @property
    def duration(self) -> float:
        return self.delay


# =============================================================================
# Factory Functions
# =============================================================================

def make_sinc_pulse(
    flip_angle: float,
    duration: float = 3e-3,
    slice_thickness: float = None,
    apodization: float = 0.5,
    time_bw_product: float = 4,
    center_pos: float = 0.5,
    system: SystemLimits = None,
    use: str = 'excitation',
    return_gz: bool = True
) -> Union[RFPulse, Tuple[RFPulse, Trapezoid, Trapezoid]]:
    """
    Create a slice-selective sinc pulse with optional gradient
    
    Args:
        flip_angle: Flip angle in radians
        duration: Pulse duration (s)
        slice_thickness: Slice thickness (m), None for non-selective
        apodization: Hanning window apodization factor (0-1)
        time_bw_product: Time-bandwidth product
        center_pos: Position of pulse center (0-1)
        system: System limits
        use: 'excitation', 'refocusing', or 'inversion'
        return_gz: If True and slice_thickness given, return (rf, gz, gz_reph)
    
    Returns:
        RFPulse, or (RFPulse, gz, gz_reph) if slice_thickness given and return_gz=True
    """
    if system is None:
        system = SystemLimits()
    
    # Calculate bandwidth
    bw = time_bw_product / duration
    
    # Create time vector
    n_samples = int(np.round(duration / system.rf_raster_time))
    t = np.arange(n_samples) * system.rf_raster_time
    
    # Create sinc pulse
    t_centered = t - duration * center_pos
    signal = np.sinc(bw * t_centered)
    
    # Apply Hanning window apodization
    if apodization > 0:
        window = (1 - apodization) + apodization * np.cos(2 * np.pi * t_centered / duration)
        signal = signal * window
    
    # Scale for flip angle
    signal = signal / np.sum(signal) * flip_angle / (2 * np.pi * system.rf_raster_time)
    
    # Calculate RF center (time of peak)
    rf_center = duration * center_pos
    
    rf = RFPulse(
        signal=signal.astype(complex),
        t=t,
        delay=system.rf_dead_time,
        dead_time=system.rf_dead_time,
        ringdown_time=system.rf_ringdown_time,
        shape_dur=duration,
        use=use,
        center=rf_center
    )
    
    if slice_thickness is not None and return_gz:
        # Create slice select gradient
        gz_amplitude = bw / slice_thickness
        
        # Check against system limits
        if abs(gz_amplitude) > system.max_grad:
            raise ValueError(f"Slice select gradient exceeds max_grad: {gz_amplitude} > {system.max_grad}")
        
        gz_rise_time = abs(gz_amplitude) / system.max_slew
        gz_rise_time = np.ceil(gz_rise_time / system.grad_raster_time) * system.grad_raster_time
        
        gz = Trapezoid(
            channel='z',
            amplitude=gz_amplitude,
            rise_time=gz_rise_time,
            flat_time=duration,
            fall_time=gz_rise_time,
            delay=system.rf_dead_time - gz_rise_time,
            area=gz_amplitude * (duration + gz_rise_time),
            flat_area=gz_amplitude * duration
        )
        
        # Create rephasing gradient
        gz_reph_area = -gz.area / 2
        gz_reph = make_trapezoid('z', area=gz_reph_area, system=system)
        
        return rf, gz, gz_reph
    
    return rf


def make_block_pulse(
    flip_angle: float,
    duration: float = 500e-6,
    bandwidth: float = None,
    system: SystemLimits = None,
    use: str = 'excitation'
) -> RFPulse:
    """
    Create a rectangular (hard) RF pulse
    
    Args:
        flip_angle: Flip angle in radians
        duration: Pulse duration (s)
        bandwidth: Bandwidth (Hz), computed from duration if None
        system: System limits
        use: 'excitation', 'refocusing', or 'inversion'
    
    Returns:
        RFPulse object
    """
    if system is None:
        system = SystemLimits()
    
    n_samples = max(1, int(np.round(duration / system.rf_raster_time)))
    t = np.arange(n_samples) * system.rf_raster_time
    
    # Constant amplitude for block pulse
    amplitude = flip_angle / (2 * np.pi * duration)
    signal = np.ones(n_samples, dtype=complex) * amplitude
    
    return RFPulse(
        signal=signal,
        t=t,
        delay=system.rf_dead_time,
        dead_time=system.rf_dead_time,
        ringdown_time=system.rf_ringdown_time,
        shape_dur=duration,
        use=use,
        center=duration / 2
    )


def make_trapezoid(
    channel: str,
    area: float = None,
    flat_area: float = None,
    amplitude: float = None,
    duration: float = None,
    flat_time: float = None,
    rise_time: float = None,
    fall_time: float = None,
    delay: float = 0,
    system: SystemLimits = None
) -> Trapezoid:
    """
    Create a trapezoidal gradient
    
    Multiple calling conventions supported:
    - area + duration: Triangle/trapezoid to achieve area in given time
    - flat_area + flat_time: Trapezoid with specific flat portion
    - amplitude + flat_time: Direct specification
    - area only: Minimum duration triangle
    
    Args:
        channel: 'x', 'y', or 'z'
        area: Total gradient area (Hz/m * s)
        flat_area: Area during flat portion (Hz/m * s)
        amplitude: Gradient amplitude (Hz/m)
        duration: Total duration (s)
        flat_time: Flat top duration (s)
        rise_time: Rise time (s), computed from slew if None
        fall_time: Fall time (s), defaults to rise_time
        delay: Delay before gradient (s)
        system: System limits
    
    Returns:
        Trapezoid object
    """
    if system is None:
        system = SystemLimits()
    
    # Determine rise/fall times from slew rate if not specified
    def calc_rise_time(amp):
        rt = abs(amp) / system.max_slew
        return np.ceil(rt / system.grad_raster_time) * system.grad_raster_time
    
    if flat_area is not None and flat_time is not None:
        # Specify flat portion directly
        amplitude = flat_area / flat_time
        if abs(amplitude) > system.max_grad:
            raise ValueError(f"Gradient amplitude {amplitude} exceeds max_grad {system.max_grad}")
        rise_time = rise_time or calc_rise_time(amplitude)
        fall_time = fall_time or rise_time
        area = amplitude * (flat_time + (rise_time + fall_time) / 2)
        
    elif area is not None and duration is not None:
        # Area in given duration - may be triangle or trapezoid
        rise_time = rise_time or np.ceil(duration / 4 / system.grad_raster_time) * system.grad_raster_time
        fall_time = fall_time or rise_time
        flat_time = duration - rise_time - fall_time
        
        if flat_time < 0:
            # Triangle gradient
            flat_time = 0
            rise_time = duration / 2
            fall_time = duration / 2
        
        amplitude = area / (flat_time + (rise_time + fall_time) / 2)
        
        if abs(amplitude) > system.max_grad:
            raise ValueError(f"Gradient amplitude {amplitude} exceeds max_grad {system.max_grad}")
        
        flat_area = amplitude * flat_time
        
    elif area is not None:
        # Minimum duration for given area
        # Use maximum slew rate to minimize time
        amplitude = np.sqrt(abs(area) * system.max_slew)
        if amplitude > system.max_grad:
            amplitude = system.max_grad
        amplitude = np.sign(area) * amplitude
        
        rise_time = calc_rise_time(amplitude)
        fall_time = rise_time
        
        # Calculate required flat time
        ramp_area = amplitude * rise_time / 2 * 2  # Both ramps
        remaining_area = abs(area) - abs(ramp_area)
        
        if remaining_area > 0:
            flat_time = remaining_area / abs(amplitude)
            flat_time = np.ceil(flat_time / system.grad_raster_time) * system.grad_raster_time
        else:
            flat_time = 0
        
        flat_area = amplitude * flat_time
        area = amplitude * (flat_time + rise_time)  # Recalculate actual area
        
    elif amplitude is not None and flat_time is not None:
        # Direct specification
        rise_time = rise_time or calc_rise_time(amplitude)
        fall_time = fall_time or rise_time
        flat_area = amplitude * flat_time
        area = amplitude * (flat_time + (rise_time + fall_time) / 2)
        
    else:
        raise ValueError("Must specify: (area), (area+duration), (flat_area+flat_time), or (amplitude+flat_time)")
    
    return Trapezoid(
        channel=channel,
        amplitude=amplitude,
        rise_time=rise_time,
        flat_time=flat_time,
        fall_time=fall_time,
        delay=delay,
        area=area,
        flat_area=flat_area
    )


def make_arbitrary_grad(
    channel: str,
    waveform: np.ndarray,
    system: SystemLimits = None,
    delay: float = 0
) -> ArbitraryGradient:
    """
    Create an arbitrary gradient waveform
    
    Args:
        channel: 'x', 'y', or 'z'
        waveform: Gradient amplitude array (Hz/m)
        system: System limits
        delay: Delay before gradient (s)
    
    Returns:
        ArbitraryGradient object
    """
    if system is None:
        system = SystemLimits()
    
    t = np.arange(len(waveform)) * system.grad_raster_time
    
    # Check amplitude limit
    if np.max(np.abs(waveform)) > system.max_grad:
        raise ValueError(f"Gradient exceeds max_grad")
    
    # Check slew rate limit
    slew = np.diff(waveform) / system.grad_raster_time
    if np.max(np.abs(slew)) > system.max_slew:
        raise ValueError(f"Gradient slew rate exceeds max_slew")
    
    return ArbitraryGradient(
        channel=channel,
        waveform=waveform,
        t=t,
        delay=delay
    )


def make_adc(
    num_samples: int,
    duration: float = None,
    dwell: float = None,
    delay: float = 0,
    system: SystemLimits = None
) -> ADC:
    """
    Create an ADC readout event
    
    Args:
        num_samples: Number of samples to acquire
        duration: Total readout duration (s)
        dwell: Dwell time per sample (s)
        delay: Delay before first sample (s)
        system: System limits
    
    Returns:
        ADC object
    """
    if system is None:
        system = SystemLimits()
    
    if duration is not None:
        dwell = duration / num_samples
    elif dwell is None:
        raise ValueError("Must specify either duration or dwell")
    
    return ADC(
        num_samples=num_samples,
        dwell=dwell,
        delay=delay,
        duration=num_samples * dwell,
        dead_time=system.adc_dead_time
    )


def make_delay(delay: float) -> Delay:
    """Create a delay event"""
    return Delay(delay=delay)


# =============================================================================
# Utility Functions
# =============================================================================

def calc_duration(*events) -> float:
    """
    Calculate the duration of one or more events
    
    Args:
        *events: One or more event objects
    
    Returns:
        Maximum duration across all events
    """
    duration = 0
    for event in events:
        if event is None:
            continue
        if isinstance(event, RFPulse):
            dur = event.delay + event.shape_dur + event.ringdown_time
        elif isinstance(event, Trapezoid):
            dur = event.delay + event.rise_time + event.flat_time + event.fall_time
        elif isinstance(event, ArbitraryGradient):
            dur = event.duration
        elif isinstance(event, ADC):
            dur = event.delay + event.num_samples * event.dwell + event.dead_time
        elif isinstance(event, Delay):
            dur = event.delay
        else:
            dur = getattr(event, 'duration', 0)
        duration = max(duration, dur)
    return duration


def calc_rf_center(rf: RFPulse) -> float:
    """Calculate the center time of an RF pulse (time of peak power)"""
    return rf.center
