# MRI-seq

A lightweight Python library for creating and visualizing MRI pulse sequences with animated k-space trajectories.

![SE-EPI k-space](output/se_epi_64_kspace.mp4)

## Features

- **Modular pulse sequence building blocks**: RF pulses, trapezoidal gradients, arbitrary gradients, Acq readouts
- **Pre-built sequences**: Spin Echo EPI, Archimedean Spiral
- **Animated k-space visualization**: Watch the k-space trajectory fill in real-time
- **Timing diagrams**: Publication-ready sequence diagrams with RF, gradient, and ADC waveforms

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MRI-seq.git
cd MRI-seq

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy matplotlib
```

## Quick Start

### Command Line

```bash
# Generate Spin Echo EPI sequence
python run.py se_epi --nx 64

# Generate Spiral sequence  
python run.py spiral --nx 64

# Options
python run.py se_epi --fov 256 --nx 128 --ny 128 --te 60
python run.py spiral --no-show --no-save
```

### Python API

```python
from mr.sequences import create_se_epi, create_spiral
from mr.animator import SequenceAnimator

# Create a Spin Echo EPI sequence
seq = create_se_epi(fov=256e-3, nx=64, ny=64, te=60e-3)
print(f"Duration: {seq.duration()*1000:.1f} ms")

# Visualize
animator = SequenceAnimator(seq)
animator.create_static_plot("timing.png")
animator.create_kspace_2d_animation("kspace.mp4", fps=50)
```

### Build Custom Sequences

```python
from mr import Sequence, SystemLimits
from mr import make_sinc_pulse, make_trapezoid, make_adc

sys = SystemLimits()
seq = Sequence(sys)

# 90° excitation with slice selection
rf, gz, gz_reph = make_sinc_pulse(
    flip_angle=np.pi/2,
    slice_thickness=3e-3,
    system=sys
)

# Readout gradient
gx = make_trapezoid('x', flat_area=64/0.256, flat_time=3.2e-3, system=sys)
adc = make_adc(64, duration=gx.flat_time, delay=gx.rise_time)

# Build sequence
seq.add_block(rf, gz)
seq.add_block(gz_reph)
seq.add_block(gx, adc)
```

## Project Structure

```
MRI-seq/
├── run.py              # CLI demo script
├── mr/
│   ├── __init__.py     # Package exports
│   ├── primitives.py   # Core building blocks (RF, gradients, ADC)
│   ├── sequence.py     # Sequence container
│   ├── animator.py     # Visualization & animation
│   └── sequences/
│       ├── se_epi.py   # Spin Echo EPI implementation
│       └── spiral.py   # Archimedean spiral implementation
└── output/             # Generated figures and animations
```

## Sequences

### Spin Echo EPI (SE-EPI)
- 90° sinc excitation → 180° refocusing → EPI readout
- Zigzag Cartesian k-space trajectory
- Configurable FOV, matrix size, TE

### Archimedean Spiral
- Single-shot spiral-out from k-space center
- Time-optimal gradient design (slew-rate limited)
- Based on pulseq `writeSpiral.m`

## References

- [Pulseq](https://pulseq.github.io/) - Open source pulse sequence framework
- Layton et al. "Pulseq: A rapid and hardware-independent pulse sequence prototyping framework" MRM 2017

## License

MIT License
