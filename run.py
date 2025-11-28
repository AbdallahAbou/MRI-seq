#!/usr/bin/env python3
"""
MRI Sequence Demo

Generate and visualize MRI pulse sequences with k-space trajectory animations.

Usage:
    python run.py <sequence_type> [options]
    
Sequences:
    spiral   - Spiral (Archimedean, starts from center)
    se_epi   - Spin Echo EPI (zigzag Cartesian)

Examples:
    python run.py spiral
    python run.py se_epi --fov 220 --nx 128
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add mr package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mr.sequences import create_se_epi, create_spiral
from mr.animator import SequenceAnimator


SEQUENCES = {
    'spiral': {
        'func': create_spiral,
        'description': 'Spiral (Archimedean spiral from center outward)',
        'defaults': {'flip': 90}
    },
    'se_epi': {
        'func': create_se_epi,
        'description': 'Spin Echo EPI (zigzag Cartesian trajectory)',
        'defaults': {'te': 60}
    }
}


def main():
    parser = argparse.ArgumentParser(
        description='MRI Sequence Demo - Generate and visualize sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join([f"  {k:10} - {v['description']}" for k, v in SEQUENCES.items()])
    )
    
    parser.add_argument('sequence', choices=SEQUENCES.keys(),
                        help='Sequence type')
    parser.add_argument('--fov', type=float, default=256,
                        help='FOV in mm (default: 256)')
    parser.add_argument('--nx', type=int, default=64,
                        help='Matrix size (default: 64)')
    parser.add_argument('--ny', type=int, default=None,
                        help='Phase encodes (default: same as nx)')
    parser.add_argument('--thickness', type=float, default=3,
                        help='Slice thickness in mm (default: 3)')
    parser.add_argument('--te', type=float, default=None,
                        help='Echo time in ms')
    parser.add_argument('--tr', type=float, default=None,
                        help='Repetition time in ms')
    parser.add_argument('--flip', type=float, default=None,
                        help='Flip angle in degrees')
    parser.add_argument('--output-dir', '-o', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--no-show', action='store_true',
                        help='Skip interactive display')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving files')
    parser.add_argument('--fps', type=int, default=50,
                        help='Animation FPS (default: 50)')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Animation duration in seconds (default: 10)')
    
    args = parser.parse_args()
    
    # Header
    print(f"\n{'='*60}")
    print(f"  MRI Sequence Demo: {args.sequence.upper()}")
    print(f"  {SEQUENCES[args.sequence]['description']}")
    print(f"{'='*60}\n")
    
    # Get defaults for this sequence
    defaults = SEQUENCES[args.sequence]['defaults']
    
    # Build parameters
    fov = args.fov * 1e-3
    thickness = args.thickness * 1e-3
    ny = args.ny or args.nx
    
    # Create sequence
    print(f"Creating sequence...")
    seq_func = SEQUENCES[args.sequence]['func']
    
    if args.sequence == 'spiral':
        seq = seq_func(
            fov=fov,
            nx=args.nx,
            thickness=thickness,
            flip_angle=np.deg2rad(args.flip or defaults.get('flip', 90))
        )
    elif args.sequence == 'se_epi':
        seq = seq_func(
            fov=fov,
            nx=args.nx,
            ny=ny,
            thickness=thickness,
            te=(args.te or defaults.get('te', 60)) * 1e-3
        )
    
    print(f"  {seq}")
    print(f"  FOV: {args.fov}mm, Matrix: {args.nx}, Slice: {args.thickness}mm")
    
    # Create animator
    animator = SequenceAnimator(seq, time_scale='ms')
    
    # Output paths
    os.makedirs(args.output_dir, exist_ok=True)
    base = f"{args.sequence}_{args.nx}"
    timing_file = os.path.join(args.output_dir, f"{base}_timing.png")
    kspace_file = os.path.join(args.output_dir, f"{base}_kspace.mp4")
    
    # Save files
    if not args.no_save:
        print(f"\nSaving timing diagram: {timing_file}")
        animator.create_static_plot(timing_file)
        plt.close()  # Close the figure after saving
        
        print(f"Saving k-space animation: {kspace_file}")
        animator.create_kspace_2d_animation(
            save_path=kspace_file,
            fps=args.fps,
            duration=args.duration
        )
        plt.close()  # Close the figure after saving
    
    # Show interactive
    if not args.no_show:
        print("\nOpening interactive animation window...")
        print("(Close the window to exit)")
        # IMPORTANT: Keep reference to animation object to prevent garbage collection
        ani = animator.create_kspace_2d_animation(
            save_path=None,
            fps=args.fps,
            duration=args.duration
        )
        plt.show()
    
    # Summary
    print(f"\n{'='*60}")
    print("  Complete!")
    if not args.no_save:
        print(f"\n  Output files:")
        print(f"    {timing_file}")
        print(f"    {kspace_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
