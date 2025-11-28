"""
MR Sequence Container

This module provides the Sequence class for building and manipulating
MRI pulse sequences.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from .primitives import (
    SystemLimits, RFPulse, Trapezoid, ArbitraryGradient, ADC, Delay,
    calc_duration
)


class Sequence:
    """
    MRI pulse sequence container
    
    Stores sequence blocks and provides methods for manipulation,
    waveform extraction, and k-space trajectory calculation.
    
    Example:
        from mr import Sequence, SystemLimits, make_sinc_pulse, make_trapezoid
        
        sys = SystemLimits()
        seq = Sequence(sys)
        
        rf, gz, gz_reph = make_sinc_pulse(np.pi/2, slice_thickness=3e-3, system=sys)
        seq.add_block(rf, gz)
        seq.add_block(gz_reph)
        
        print(f"Duration: {seq.duration()*1000:.2f} ms")
    """
    
    def __init__(self, system: SystemLimits = None):
        """
        Initialize a sequence
        
        Args:
            system: System limits (uses defaults if None)
        """
        self.system = system or SystemLimits()
        self.blocks: List[Dict[str, Any]] = []
        self.definitions: Dict[str, Any] = {}
    
    def add_block(self, *events):
        """
        Add a block with one or more simultaneous events
        
        Args:
            *events: RFPulse, Trapezoid, ArbitraryGradient, ADC, or Delay objects
        """
        block = {
            'rf': None,
            'gx': None,
            'gy': None,
            'gz': None,
            'adc': None,
            'delay': None
        }
        
        for event in events:
            if event is None:
                continue
            if isinstance(event, RFPulse):
                block['rf'] = event
            elif isinstance(event, (Trapezoid, ArbitraryGradient)):
                block[f'g{event.channel}'] = event
            elif isinstance(event, ADC):
                block['adc'] = event
            elif isinstance(event, Delay):
                block['delay'] = event
        
        block['duration'] = calc_duration(*events)
        self.blocks.append(block)
    
    def set_definition(self, key: str, value: Any):
        """Set a sequence definition (metadata)"""
        self.definitions[key] = value
    
    def get_definition(self, key: str, default: Any = None) -> Any:
        """Get a sequence definition"""
        return self.definitions.get(key, default)
    
    def duration(self) -> float:
        """Get total sequence duration in seconds"""
        return sum(b['duration'] for b in self.blocks)
    
    def num_blocks(self) -> int:
        """Get number of blocks"""
        return len(self.blocks)
    
    def get_block(self, index: int) -> Dict[str, Any]:
        """Get a specific block by index"""
        return self.blocks[index]
    
    def get_block_waveforms(self) -> List[Dict[str, Any]]:
        """
        Get waveforms organized by block for animation
        
        Returns:
            List of dictionaries, one per block, containing:
            - t_start, t_end: Block timing
            - rf, gx, gy, gz, adc: Event waveform data
        """
        block_data = []
        t_offset = 0
        
        for block in self.blocks:
            data = {
                't_start': t_offset,
                't_end': t_offset + block['duration'],
                'duration': block['duration'],
                'rf': None,
                'gx': None,
                'gy': None,
                'gz': None,
                'adc': None
            }
            
            # RF
            if block['rf'] is not None:
                rf = block['rf']
                data['rf'] = {
                    't': t_offset + rf.delay + rf.t,
                    'magnitude': np.abs(rf.signal),
                    'phase': np.angle(rf.signal),
                    'use': rf.use
                }
            
            # Gradients
            for channel in ['x', 'y', 'z']:
                grad = block[f'g{channel}']
                if grad is not None:
                    if isinstance(grad, Trapezoid):
                        t_grad = t_offset + grad.delay + np.array([
                            0, grad.rise_time,
                            grad.rise_time + grad.flat_time,
                            grad.rise_time + grad.flat_time + grad.fall_time
                        ])
                        amp = np.array([0, grad.amplitude, grad.amplitude, 0])
                    else:  # ArbitraryGradient
                        t_grad = t_offset + grad.delay + grad.t
                        amp = grad.waveform
                    data[f'g{channel}'] = {'t': t_grad, 'amplitude': amp}
            
            # ADC
            if block['adc'] is not None:
                adc = block['adc']
                data['adc'] = {
                    't_start': t_offset + adc.delay,
                    't_end': t_offset + adc.delay + adc.num_samples * adc.dwell,
                    'num_samples': adc.num_samples
                }
            
            block_data.append(data)
            t_offset += block['duration']
        
        return block_data
    
    def calculate_kspace(self, dt: float = 10e-6) -> Dict[str, np.ndarray]:
        """
        Calculate k-space trajectory from gradient waveforms
        
        Args:
            dt: Time step for integration (s)
        
        Returns:
            Dictionary with 't', 'kx', 'ky', 'kz', 'adc_mask' arrays
        """
        total_duration = self.duration()
        n_points = int(np.ceil(total_duration / dt))
        
        t = np.arange(n_points) * dt
        gx = np.zeros(n_points)
        gy = np.zeros(n_points)
        gz = np.zeros(n_points)
        adc_mask = np.zeros(n_points, dtype=bool)
        
        t_offset = 0
        
        for block in self.blocks:
            # Sample gradients at each time point
            for channel, g_arr in [('x', gx), ('y', gy), ('z', gz)]:
                grad = block[f'g{channel}']
                if grad is None:
                    continue
                
                if isinstance(grad, Trapezoid):
                    # Sample trapezoid
                    for i, ti in enumerate(t):
                        if t_offset <= ti < t_offset + block['duration']:
                            local_t = ti - t_offset - grad.delay
                            if local_t < 0:
                                amp = 0
                            elif local_t < grad.rise_time:
                                amp = grad.amplitude * local_t / grad.rise_time
                            elif local_t < grad.rise_time + grad.flat_time:
                                amp = grad.amplitude
                            elif local_t < grad.rise_time + grad.flat_time + grad.fall_time:
                                amp = grad.amplitude * (1 - (local_t - grad.rise_time - grad.flat_time) / grad.fall_time)
                            else:
                                amp = 0
                            g_arr[i] = amp
                else:  # ArbitraryGradient
                    for i, ti in enumerate(t):
                        if t_offset <= ti < t_offset + block['duration']:
                            local_t = ti - t_offset - grad.delay
                            if 0 <= local_t <= grad.t[-1]:
                                amp = np.interp(local_t, grad.t, grad.waveform)
                                g_arr[i] = amp
            
            # Mark ADC samples
            if block['adc'] is not None:
                adc = block['adc']
                adc_start = t_offset + adc.delay
                adc_end = adc_start + adc.num_samples * adc.dwell
                adc_mask[(t >= adc_start) & (t < adc_end)] = True
            
            t_offset += block['duration']
        
        # Integrate gradients to get k-space position
        kx = np.cumsum(gx) * dt
        ky = np.cumsum(gy) * dt
        kz = np.cumsum(gz) * dt
        
        return {
            't': t,
            'kx': kx,
            'ky': ky,
            'kz': kz,
            'gx': gx,
            'gy': gy,
            'gz': gz,
            'adc_mask': adc_mask
        }
    
    def plot_kspace_2d(self, ax=None, show_trajectory: bool = True, show_samples: bool = True):
        """
        Plot 2D k-space trajectory (kx vs ky)
        
        Args:
            ax: Matplotlib axis (creates new figure if None)
            show_trajectory: Show full trajectory line
            show_samples: Show ADC sample points
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        kspace = self.calculate_kspace()
        
        if show_trajectory:
            ax.plot(kspace['kx'], kspace['ky'], 'b-', alpha=0.5, linewidth=0.5)
        
        if show_samples:
            ax.scatter(kspace['kx'][kspace['adc_mask']], 
                      kspace['ky'][kspace['adc_mask']], 
                      c='red', s=2, zorder=5)
        
        ax.set_xlabel('kx (1/m)')
        ax.set_ylabel('ky (1/m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('K-space Trajectory')
        
        return ax
    
    def __repr__(self) -> str:
        return f"Sequence({len(self.blocks)} blocks, {self.duration()*1000:.2f} ms)"
