"""
MRI Sequence Animation Module

Creates animated visualizations of MRI pulse sequences using matplotlib.
Supports both real-time animation and video export.

Based on the original sequence_animator.py style with enhanced features.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from typing import List, Dict, Optional, Tuple
import warnings


class SequenceAnimator:
    """
    Animated visualization of MRI pulse sequences
    
    Creates a multi-panel plot showing:
    - RF magnitude and phase
    - Gradient waveforms (Gs=slice, Gf=frequency/readout, Gp=phase)
    - ADC/Acq acquisition windows
    - Optional k-space trajectory
    
    Example:
        from mr.sequences import create_se_epi, create_spiral
        from mr.animator import SequenceAnimator
        
        seq = create_spiral()
        animator = SequenceAnimator(seq)
        animator.create_kspace_2d_animation('spiral.mp4')
    """
    
    def __init__(self, sequence, 
                 figsize: Tuple[int, int] = (14, 10),
                 time_scale: str = 'ms'):
        """
        Initialize the animator
        
        Args:
            sequence: Sequence object to animate
            figsize: Figure size (width, height)
            time_scale: Time scale for display ('s', 'ms', 'us')
        """
        self.seq = sequence
        self.figsize = figsize
        self.time_scale = time_scale
        
        # Time scaling factor
        self.time_factor = {'s': 1, 'ms': 1e3, 'us': 1e6}[time_scale]
        self.time_label = f'Time ({time_scale})'
        
        # Get block-organized waveform data
        self.block_data = sequence.get_block_waveforms()
        self.total_duration = sequence.duration()
        
        # Pre-compute all waveforms for the full sequence
        self._precompute_waveforms()
        
        # Animation state
        self.current_time = 0
        self.ani = None
    
    def _precompute_waveforms(self):
        """Pre-compute continuous waveforms for plotting"""
        # Create time array with fine resolution
        dt = 1e-6  # 1 us resolution
        self.t_full = np.arange(0, self.total_duration + dt, dt)
        n_points = len(self.t_full)
        
        # Initialize waveform arrays
        self.rf_mag_full = np.zeros(n_points)
        self.rf_phase_full = np.zeros(n_points)
        self.gx_full = np.zeros(n_points)
        self.gy_full = np.zeros(n_points)
        self.gz_full = np.zeros(n_points)
        self.adc_full = np.zeros(n_points)
        
        # Fill in waveforms from block data
        for block in self.block_data:
            t_start_idx = int(block['t_start'] / dt)
            t_end_idx = min(int(block['t_end'] / dt), n_points)
            
            # RF
            if block['rf'] is not None:
                rf = block['rf']
                for i, (t, mag, phase) in enumerate(zip(rf['t'], rf['magnitude'], rf['phase'])):
                    idx = int(t / dt)
                    if 0 <= idx < n_points:
                        self.rf_mag_full[idx] = mag
                        self.rf_phase_full[idx] = phase
            
            # Gradients - interpolate
            for channel, arr in [('gx', self.gx_full), ('gy', self.gy_full), ('gz', self.gz_full)]:
                grad = block[channel]
                if grad is not None:
                    t_grad = grad['t']
                    amp = grad['amplitude']
                    # Interpolate to fill array
                    for i in range(len(t_grad) - 1):
                        idx_start = int(t_grad[i] / dt)
                        idx_end = int(t_grad[i+1] / dt)
                        if idx_start < n_points and idx_end > 0:
                            idx_start = max(0, idx_start)
                            idx_end = min(n_points, idx_end)
                            t_segment = self.t_full[idx_start:idx_end]
                            if len(t_segment) > 0:
                                interp_vals = np.interp(t_segment, t_grad, amp)
                                arr[idx_start:idx_end] = interp_vals
            
            # ADC
            if block['adc'] is not None:
                adc = block['adc']
                adc_start_idx = int(adc['t_start'] / dt)
                adc_end_idx = int(adc['t_end'] / dt)
                if 0 <= adc_start_idx < n_points:
                    self.adc_full[adc_start_idx:min(adc_end_idx, n_points)] = 1
    
    def _add_rf_labels(self, ax):
        """Add labels for RF pulses (90°, 180°)"""
        rf_events = []
        for block in self.block_data:
            if block['rf'] is not None:
                t_center = np.mean(block['rf']['t'])
                use = block['rf']['use']
                rf_events.append((t_center, use))
        
        for t, use in rf_events:
            label = '90°' if use == 'excitation' else '180°'
            ax.annotate(label, xy=(t * self.time_factor, ax.get_ylim()[1] * 0.8),
                       fontsize=10, ha='center', fontweight='bold',
                       color='darkblue')
    
    def create_static_plot(self, filename: str = None, figsize: tuple = None) -> plt.Figure:
        """
        Create a static plot of the entire sequence
        
        Args:
            filename: If provided, save to this file
            figsize: Optional figure size override
        
        Returns:
            matplotlib Figure object
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)
        seq_name = self.seq.get_definition('Name', 'MRI Sequence')
        fig.suptitle(f'{seq_name.upper()} Sequence', fontsize=14, fontweight='bold')
        
        t_scaled = self.t_full * self.time_factor
        
        # RF Magnitude
        axes[0].plot(t_scaled, self.rf_mag_full, 'b-', linewidth=1)
        axes[0].fill_between(t_scaled, 0, self.rf_mag_full, alpha=0.3, color='blue')
        axes[0].set_ylabel('RF Mag\n(Hz)', fontsize=9)
        axes[0].set_ylim(bottom=0)
        axes[0].grid(True, alpha=0.3)
        self._add_rf_labels(axes[0])
        
        # RF Phase
        axes[1].plot(t_scaled, self.rf_phase_full, 'purple', linewidth=1)
        axes[1].set_ylabel('RF Phase\n(rad)', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        # Gf (Frequency/Readout - X axis)
        axes[2].plot(t_scaled, self.gx_full, 'r-', linewidth=1)
        axes[2].fill_between(t_scaled, 0, self.gx_full, alpha=0.3, color='red')
        axes[2].set_ylabel('Gf\n(Hz/m)', fontsize=9)
        axes[2].axhline(y=0, color='k', linewidth=0.5)
        axes[2].grid(True, alpha=0.3)
        
        # Gp (Phase encode - Y axis)
        axes[3].plot(t_scaled, self.gy_full, 'g-', linewidth=1)
        axes[3].fill_between(t_scaled, 0, self.gy_full, alpha=0.3, color='green')
        axes[3].set_ylabel('Gp\n(Hz/m)', fontsize=9)
        axes[3].axhline(y=0, color='k', linewidth=0.5)
        axes[3].grid(True, alpha=0.3)
        
        # Gs (Slice select - Z axis)
        axes[4].plot(t_scaled, self.gz_full, 'b-', linewidth=1)
        axes[4].fill_between(t_scaled, 0, self.gz_full, alpha=0.3, color='blue')
        axes[4].set_ylabel('Gs\n(Hz/m)', fontsize=9)
        axes[4].axhline(y=0, color='k', linewidth=0.5)
        axes[4].grid(True, alpha=0.3)
        
        # Acq (ADC)
        axes[5].fill_between(t_scaled, 0, self.adc_full, alpha=0.7, color='orange')
        axes[5].set_ylabel('Acq', fontsize=9)
        axes[5].set_xlabel(self.time_label, fontsize=10)
        axes[5].set_ylim(0, 1.5)
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        return fig
    
    def create_animation(self, duration: float = 10.0, 
                        fps: int = 30,
                        repeat: bool = True) -> animation.FuncAnimation:
        """
        Create an animated visualization
        
        Args:
            duration: Animation duration in seconds (real time)
            fps: Frames per second
            repeat: Whether to loop the animation
        
        Returns:
            matplotlib FuncAnimation object
        """
        fig, axes = plt.subplots(6, 1, figsize=self.figsize, sharex=True)
        seq_name = self.seq.get_definition('Name', 'MRI Sequence')
        fig.suptitle(f'{seq_name.upper()} Sequence - Animation', fontsize=14, fontweight='bold')
        
        t_scaled = self.t_full * self.time_factor
        t_max = self.total_duration * self.time_factor
        
        # Initialize empty lines
        lines = []
        time_markers = []
        
        # Set up each axis (using lecture notation: Gs=slice, Gf=freq, Gp=phase)
        labels = ['RF Mag (Hz)', 'RF Phase (rad)', 'Gf (Hz/m)', 'Gp (Hz/m)', 'Gs (Hz/m)', 'Acq']
        colors = ['blue', 'purple', 'red', 'green', 'blue', 'orange']
        data = [self.rf_mag_full, self.rf_phase_full, self.gx_full, 
                self.gy_full, self.gz_full, self.adc_full]
        
        for i, (ax, label, color, d) in enumerate(zip(axes, labels, colors, data)):
            ax.set_xlim(0, t_max)
            y_min, y_max = np.min(d), np.max(d)
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Background (faded full sequence)
            ax.plot(t_scaled, d, color=color, alpha=0.2, linewidth=0.5)
            
            # Animated line
            line, = ax.plot([], [], color=color, linewidth=1.5)
            lines.append(line)
            
            # Time marker (vertical line)
            marker = ax.axvline(x=0, color='red', linewidth=1, linestyle='--', alpha=0.7)
            time_markers.append(marker)
            
            if i >= 2 and i <= 4:  # Gradient axes
                ax.axhline(y=0, color='k', linewidth=0.5)
        
        axes[-1].set_xlabel(self.time_label, fontsize=10)
        
        n_frames = int(duration * fps)
        
        def init():
            for line in lines:
                line.set_data([], [])
            for marker in time_markers:
                marker.set_xdata([0])
            return lines + time_markers
        
        def animate(frame):
            current_t = (frame / n_frames) * self.total_duration
            current_idx = int(current_t / (self.t_full[1] - self.t_full[0]))
            current_idx = min(current_idx, len(self.t_full) - 1)
            
            t_plot = t_scaled[:current_idx+1]
            for i, (line, d) in enumerate(zip(lines, data)):
                line.set_data(t_plot, d[:current_idx+1])
            
            current_t_scaled = current_t * self.time_factor
            for marker in time_markers:
                marker.set_xdata([current_t_scaled])
            
            return lines + time_markers
        
        self.ani = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=n_frames, interval=1000/fps,
            blit=True, repeat=repeat
        )
        
        plt.tight_layout()
        return self.ani
    
    def create_kspace_animation(self, duration: float = 10.0,
                                fps: int = 30) -> animation.FuncAnimation:
        """
        Create animation showing k-space trajectory alongside sequence
        
        Args:
            duration: Animation duration in seconds
            fps: Frames per second
        
        Returns:
            matplotlib FuncAnimation object
        """
        fig = plt.figure(figsize=(16, 8))
        
        # Create subplot layout
        gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
        ax_gx = fig.add_subplot(gs[0, 0])
        ax_gy = fig.add_subplot(gs[1, 0], sharex=ax_gx)
        ax_adc = fig.add_subplot(gs[2, 0], sharex=ax_gx)
        ax_kspace = fig.add_subplot(gs[:, 1])
        
        seq_name = self.seq.get_definition('Name', 'MRI Sequence')
        fig.suptitle(f'{seq_name.upper()}: Sequence and K-space Trajectory', fontsize=14, fontweight='bold')
        
        t_scaled = self.t_full * self.time_factor
        t_max = self.total_duration * self.time_factor
        
        # Calculate k-space trajectory
        dt = self.t_full[1] - self.t_full[0]
        kx = np.cumsum(self.gx_full) * dt
        ky = np.cumsum(self.gy_full) * dt
        
        # Find ADC sample points
        adc_indices = np.where(self.adc_full > 0)[0]
        
        # Set up gradient axes (Gf=frequency/readout, Gp=phase)
        for ax, data, color, label in [
            (ax_gx, self.gx_full, 'red', 'Gf (Hz/m)'),
            (ax_gy, self.gy_full, 'green', 'Gp (Hz/m)'),
        ]:
            ax.set_xlim(0, t_max)
            ax.set_ylabel(label, fontsize=9)
            ax.plot(t_scaled, data, color=color, alpha=0.2, linewidth=0.5)
            ax.fill_between(t_scaled, 0, data, alpha=0.1, color=color)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.grid(True, alpha=0.3)
        
        # Acq axis
        ax_adc.set_xlim(0, t_max)
        ax_adc.set_ylabel('Acq', fontsize=9)
        ax_adc.set_xlabel(self.time_label, fontsize=10)
        ax_adc.fill_between(t_scaled, 0, self.adc_full, alpha=0.2, color='orange')
        ax_adc.set_ylim(0, 1.5)
        ax_adc.grid(True, alpha=0.3)
        
        # K-space axis (kf=frequency direction, kp=phase direction)
        kx_range = max(np.max(np.abs(kx)), 1)
        ky_range = max(np.max(np.abs(ky)), 1)
        ax_kspace.set_xlim(-kx_range * 1.1, kx_range * 1.1)
        ax_kspace.set_ylim(-ky_range * 1.1, ky_range * 1.1)
        ax_kspace.set_xlabel('kf (1/m)', fontsize=10)
        ax_kspace.set_ylabel('kp (1/m)', fontsize=10)
        ax_kspace.set_title('K-space Trajectory', fontsize=11)
        ax_kspace.grid(True, alpha=0.3)
        ax_kspace.set_aspect('equal')
        ax_kspace.axhline(y=0, color='k', linewidth=0.5)
        ax_kspace.axvline(x=0, color='k', linewidth=0.5)
        
        # Initialize animated elements
        line_gx, = ax_gx.plot([], [], 'r-', linewidth=1.5)
        line_gy, = ax_gy.plot([], [], 'g-', linewidth=1.5)
        line_adc, = ax_adc.plot([], [], 'orange', linewidth=2)
        line_kspace, = ax_kspace.plot([], [], 'b-', linewidth=1, alpha=0.7)
        scatter_kspace = ax_kspace.scatter([], [], c='red', s=10, zorder=5)
        point_kspace, = ax_kspace.plot([], [], 'ko', markersize=8)
        
        # Time markers
        markers = [ax.axvline(x=0, color='red', linestyle='--', alpha=0.7) 
                   for ax in [ax_gx, ax_gy, ax_adc]]
        
        n_frames = int(duration * fps)
        
        def init():
            line_gx.set_data([], [])
            line_gy.set_data([], [])
            line_adc.set_data([], [])
            line_kspace.set_data([], [])
            scatter_kspace.set_offsets(np.empty((0, 2)))
            point_kspace.set_data([], [])
            return [line_gx, line_gy, line_adc, line_kspace, scatter_kspace, point_kspace] + markers
        
        def animate(frame):
            idx = int((frame / n_frames) * len(self.t_full))
            idx = min(idx, len(self.t_full) - 1)
            
            t_plot = t_scaled[:idx+1]
            
            line_gx.set_data(t_plot, self.gx_full[:idx+1])
            line_gy.set_data(t_plot, self.gy_full[:idx+1])
            line_adc.set_data(t_plot, self.adc_full[:idx+1])
            
            line_kspace.set_data(kx[:idx+1], ky[:idx+1])
            point_kspace.set_data([kx[idx]], [ky[idx]])
            
            acquired_idx = adc_indices[adc_indices <= idx]
            if len(acquired_idx) > 0:
                scatter_kspace.set_offsets(np.column_stack([kx[acquired_idx], ky[acquired_idx]]))
            
            current_t = t_scaled[idx]
            for marker in markers:
                marker.set_xdata([current_t])
            
            return [line_gx, line_gy, line_adc, line_kspace, scatter_kspace, point_kspace] + markers
        
        ani = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=n_frames, interval=1000/fps,
            blit=True, repeat=True
        )
        
        plt.tight_layout()
        return ani
    
    def create_kspace_2d_animation(self, 
                                   save_path: str = 'kspace.mp4',
                                   duration: float = 10.0,
                                   fps: int = 50,
                                   interval: int = 20,
                                   figsize: Tuple[int, int] = (8, 8)) -> animation.FuncAnimation:
        """
        Create a simple 2D k-space trajectory animation
        
        Shows only the k-space plot with:
        - Background grid
        - Blue line showing the path traveled
        - Red dot showing current k-space position
        - Green dots for ADC samples
        
        Args:
            save_path: Path to save animation (e.g., 'kspace.mp4')
            duration: Animation duration in seconds (real time)
            fps: Frames per second for saved video
            interval: Interval between frames in ms (for display)
            figsize: Figure size
        
        Returns:
            matplotlib FuncAnimation object
        """
        # Calculate k-space trajectory from pre-computed gradients
        dt = self.t_full[1] - self.t_full[0]
        
        kx = np.cumsum(self.gx_full) * dt
        ky = np.cumsum(self.gy_full) * dt
        
        # Calculate k-space extent
        kx_max = max(np.max(np.abs(kx)), 1)
        ky_max = max(np.max(np.abs(ky)), 1)
        k_max = max(kx_max, ky_max) * 1.1
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Set up axes
        ax.set_xlim(-k_max, k_max)
        ax.set_ylim(-k_max, k_max)
        ax.set_aspect('equal')
        
        # Grid and labels (kf=frequency, kp=phase)
        ax.grid(True, linestyle='-', alpha=0.3, color='gray')
        ax.axhline(y=0, color='black', linewidth=1)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('kf (1/m)', fontsize=12)
        ax.set_ylabel('kp (1/m)', fontsize=12)
        
        seq_name = self.seq.get_definition('Name', 'MRI Sequence')
        ax.set_title(f'K-Space Trajectory: {seq_name.upper()}', fontsize=14, fontweight='bold')
        
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        
        # Plot elements
        line_trail, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.7, label='Trajectory')
        point_current, = ax.plot([], [], 'ro', markersize=12, markeredgecolor='darkred', 
                                  markeredgewidth=2, label='Current position')
        
        adc_indices = np.where(self.adc_full > 0)[0]
        scatter_adc = ax.scatter([], [], c='lime', s=8, alpha=0.6, zorder=3, label='ADC samples')
        
        ax.legend(loc='upper right', fontsize=10)
        
        # Calculate frames
        n_frames = int(duration * fps)
        n_points = len(self.t_full)
        
        # Time text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def init():
            line_trail.set_data([], [])
            point_current.set_data([], [])
            scatter_adc.set_offsets(np.empty((0, 2)))
            time_text.set_text('')
            return line_trail, point_current, scatter_adc, time_text
        
        def animate(frame):
            idx = int((frame / n_frames) * n_points)
            idx = min(idx, n_points - 1)
            
            line_trail.set_data(kx[:idx+1], ky[:idx+1])
            point_current.set_data([kx[idx]], [ky[idx]])
            
            acquired_idx = adc_indices[adc_indices <= idx]
            if len(acquired_idx) > 0:
                scatter_adc.set_offsets(np.column_stack([kx[acquired_idx], ky[acquired_idx]]))
            
            current_time = self.t_full[idx] * 1000  # ms
            time_text.set_text(f't = {current_time:.2f} ms')
            
            return line_trail, point_current, scatter_adc, time_text
        
        ani = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=n_frames, interval=interval,
            blit=True, repeat=True
        )
        
        plt.tight_layout()
        
        # Save
        if save_path is not None:
            print(f"Saving animation to {save_path}...")
            if save_path.endswith('.gif'):
                writer = animation.PillowWriter(fps=fps)
            else:
                try:
                    writer = animation.FFMpegWriter(fps=fps, bitrate=2000, 
                                                     extra_args=['-vcodec', 'libx264'])
                except Exception as e:
                    print(f"FFmpeg error: {e}")
                    print("Falling back to GIF format...")
                    save_path = save_path.rsplit('.', 1)[0] + '.gif'
                    writer = animation.PillowWriter(fps=fps)
            
            ani.save(save_path, writer=writer, dpi=150)
            print(f"Animation saved to {save_path}")
        
        return ani
    
    def save_animation(self, filename: str, 
                       animation_type: str = 'sequence',
                       duration: float = 10.0,
                       fps: int = 30,
                       dpi: int = 150):
        """
        Save animation to file
        
        Args:
            filename: Output filename (e.g., 'animation.mp4', 'animation.gif')
            animation_type: 'sequence', 'kspace', or 'kspace_2d'
            duration: Animation duration
            fps: Frames per second
            dpi: Resolution
        """
        if animation_type == 'kspace':
            ani = self.create_kspace_animation(duration, fps)
        elif animation_type == 'kspace_2d':
            ani = self.create_kspace_2d_animation(save_path=None, duration=duration, fps=fps)
        else:
            ani = self.create_animation(duration, fps)
        
        if filename.endswith('.gif'):
            writer = animation.PillowWriter(fps=fps)
        else:
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            except:
                warnings.warn("FFmpeg not available, using Pillow for GIF output")
                filename = filename.rsplit('.', 1)[0] + '.gif'
                writer = animation.PillowWriter(fps=fps)
        
        ani.save(filename, writer=writer, dpi=dpi)
        print(f"Animation saved to {filename}")


def animate_sequence(sequence, output: str = 'sequence.mp4', **kwargs):
    """
    Convenience function to animate a sequence's k-space trajectory
    
    Args:
        sequence: Sequence object
        output: Output filename
        **kwargs: Additional arguments passed to create_kspace_2d_animation
    """
    animator = SequenceAnimator(sequence)
    animator.create_kspace_2d_animation(output, **kwargs)


def plot_sequence(sequence, output: str = None, **kwargs):
    """
    Convenience function to plot a sequence timing diagram
    
    Args:
        sequence: Sequence object
        output: Output filename (displays if None)
        **kwargs: Additional arguments passed to create_static_plot
    """
    animator = SequenceAnimator(sequence)
    fig = animator.create_static_plot(output, **kwargs)
    if output is None:
        plt.show()
    return fig


def create_timing_diagram(sequence, 
                         time_range: Optional[Tuple[float, float]] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create a publication-quality timing diagram
    
    Args:
        sequence: Sequence to plot
        time_range: Optional (start, end) time range in seconds
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    animator = SequenceAnimator(sequence, figsize=figsize)
    fig = animator.create_static_plot()
    
    if time_range is not None:
        for ax in fig.axes:
            ax.set_xlim(time_range[0] * animator.time_factor, 
                       time_range[1] * animator.time_factor)
    
    return fig
