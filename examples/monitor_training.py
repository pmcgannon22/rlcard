#!/usr/bin/env python3
'''
Live training monitor - Updates plot every 10 seconds while training runs

Usage:
    python monitor_training.py experiments/scout_improved_m1/
'''

import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_progress(log_dir, output_file='live_progress.png'):
    """Plot current training progress from CSV"""
    csv_path = Path(log_dir) / 'performance.csv'

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print(f"   Waiting for training to start...")
        return False

    try:
        # Read CSV
        df = pd.read_csv(csv_path)

        if len(df) == 0:
            print("⏳ No data yet...")
            return False

        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(df['episode'], df['reward'], linewidth=2, color='#2E86AB')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.title(f'Scout Training Progress (Latest: Episode {df["episode"].iloc[-1]}, Reward: {df["reward"].iloc[-1]:.4f})',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Add reference lines
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good performance')
        plt.axhline(y=0.7, color='gold', linestyle='--', alpha=0.5, label='Strong performance')

        plt.legend()
        plt.tight_layout()

        # Save to file
        output_path = Path(log_dir) / output_file
        plt.savefig(output_path, dpi=100)
        plt.close()

        # Print stats
        latest_episode = df['episode'].iloc[-1]
        latest_reward = df['reward'].iloc[-1]
        max_reward = df['reward'].max()
        min_reward = df['reward'].min()

        print(f"\n{'='*60}")
        print(f"📊 Training Progress (Episode {latest_episode})")
        print(f"{'='*60}")
        print(f"Latest Reward:  {latest_reward:7.4f}")
        print(f"Best Reward:    {max_reward:7.4f}")
        print(f"Worst Reward:   {min_reward:7.4f}")
        print(f"Episodes Done:  {len(df)}")
        print(f"Plot saved:     {output_path}")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"⚠️  Error reading CSV: {e}")
        return False

def monitor_live(log_dir, interval=10):
    """Monitor training progress and update plot every interval seconds"""
    print(f"🔍 Monitoring: {log_dir}")
    print(f"🔄 Updating every {interval} seconds (Ctrl+C to stop)\n")

    try:
        while True:
            plot_training_progress(log_dir)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")
        print(f"Final plot saved in: {log_dir}/live_progress.png")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python monitor_training.py <log_directory> [update_interval]")
        print("\nExample:")
        print("  python monitor_training.py experiments/scout_improved_m1/")
        print("  python monitor_training.py experiments/scout_improved_m1/ 5")
        sys.exit(1)

    log_dir = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if not os.path.exists(log_dir):
        print(f"❌ Directory not found: {log_dir}")
        sys.exit(1)

    monitor_live(log_dir, interval)
