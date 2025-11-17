#!/usr/bin/env python3
'''
Simple text-based progress viewer (no matplotlib needed)

Usage:
    python show_progress.py experiments/scout_improved_m1/
'''

import sys
from pathlib import Path

def show_progress(log_dir):
    """Show training progress in terminal"""
    csv_path = Path(log_dir) / 'performance.csv'

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print(f"   Training hasn't started or log_dir is wrong")
        return

    # Read CSV manually (no pandas needed)
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    if len(lines) <= 1:
        print("⏳ No training data yet")
        return

    # Parse data
    data = []
    for line in lines[1:]:  # Skip header
        parts = line.strip().split(',')
        if len(parts) == 2:
            episode = int(parts[0])
            reward = float(parts[1])
            data.append((episode, reward))

    if not data:
        print("⏳ No training data yet")
        return

    # Print header
    print(f"\n{'='*70}")
    print(f"📊 SCOUT TRAINING PROGRESS")
    print(f"{'='*70}")

    # Print latest stats
    latest_episode, latest_reward = data[-1]
    best_reward = max(r for _, r in data)
    worst_reward = min(r for _, r in data)
    mean_reward = sum(r for _, r in data) / len(data)

    print(f"\n📈 Summary Statistics:")
    print(f"  Episodes completed:    {latest_episode:,}")
    print(f"  Latest reward:         {latest_reward:7.4f}")
    print(f"  Best reward:           {best_reward:7.4f}")
    print(f"  Worst reward:          {worst_reward:7.4f}")
    print(f"  Mean reward:           {mean_reward:7.4f}")

    # Performance assessment
    print(f"\n🎯 Performance Assessment:")
    if latest_reward < 0:
        print(f"  Status: Learning (worse than random)")
        print(f"  Note: Agent is still exploring, this is normal early on")
    elif latest_reward < 0.3:
        print(f"  Status: Beginner (slightly better than random)")
    elif latest_reward < 0.5:
        print(f"  Status: Improving (noticeably better than random)")
    elif latest_reward < 0.7:
        print(f"  Status: Good (strong performance)")
    else:
        print(f"  Status: Excellent (beating random consistently)")

    # Print last 10 episodes
    print(f"\n📋 Last 10 Evaluations:")
    print(f"  {'Episode':>8}  {'Reward':>10}  {'Trend':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}")

    recent = data[-10:] if len(data) >= 10 else data
    for i, (ep, rew) in enumerate(recent):
        # Show trend
        if i > 0:
            prev_rew = recent[i-1][1]
            trend = "↗ improving" if rew > prev_rew else "↘ declining" if rew < prev_rew else "→ stable"
        else:
            trend = ""

        print(f"  {ep:8,}  {rew:10.4f}  {trend:>10}")

    # ASCII chart (simple)
    print(f"\n📊 Progress Chart (last 20 episodes):")
    recent_20 = data[-20:] if len(data) >= 20 else data

    if recent_20:
        min_r = min(r for _, r in recent_20)
        max_r = max(r for _, r in recent_20)
        range_r = max_r - min_r if max_r != min_r else 1

        for episode, reward in recent_20:
            # Normalize to 0-50 characters
            if range_r > 0:
                bar_len = int(((reward - min_r) / range_r) * 50)
            else:
                bar_len = 25

            bar = '█' * bar_len
            print(f"  {episode:6,} | {bar} {reward:.3f}")

    print(f"\n{'='*70}")
    print(f"💡 Tip: Run 'python examples/plot_progress.py {log_dir}' for a visual plot")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python show_progress.py <log_directory>")
        print("\nExample:")
        print("  python show_progress.py experiments/scout_improved_m1/")
        sys.exit(1)

    log_dir = sys.argv[1]
    show_progress(log_dir)
