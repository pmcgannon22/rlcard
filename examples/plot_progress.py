#!/usr/bin/env python3
'''
One-shot plot of training progress

Usage:
    python plot_progress.py experiments/scout_improved_m1/
'''

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_progress(log_dir):
    """Create a plot from current training data"""
    csv_path = Path(log_dir) / 'performance.csv'

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print(f"   Make sure training has started and log_dir is correct")
        return

    # Read CSV
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print("⏳ No training data yet")
        return

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['reward'], linewidth=2, marker='o', markersize=4, color='#2E86AB')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Scout Training Progress', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add reference lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good (50% better than random)')
    plt.axhline(y=0.7, color='gold', linestyle='--', alpha=0.5, label='Strong (70% better)')

    plt.legend()
    plt.tight_layout()

    # Save plot
    output_path = Path(log_dir) / 'current_progress.png'
    plt.savefig(output_path, dpi=150)

    # Show stats
    print(f"\n{'='*60}")
    print(f"📊 Training Statistics")
    print(f"{'='*60}")
    print(f"Episodes completed:  {df['episode'].iloc[-1]}")
    print(f"Latest reward:       {df['reward'].iloc[-1]:.4f}")
    print(f"Best reward:         {df['reward'].max():.4f}")
    print(f"Worst reward:        {df['reward'].min():.4f}")
    print(f"Mean reward:         {df['reward'].mean():.4f}")
    print(f"\n📁 Plot saved to: {output_path}")
    print(f"{'='*60}\n")

    # Try to open it
    try:
        import subprocess
        subprocess.run(['open', str(output_path)], check=False)
        print("✓ Opened plot in default viewer")
    except:
        print(f"💡 Open the plot manually: open {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_progress.py <log_directory>")
        print("\nExample:")
        print("  python plot_progress.py experiments/scout_improved_m1/")
        sys.exit(1)

    log_dir = sys.argv[1]
    plot_progress(log_dir)
