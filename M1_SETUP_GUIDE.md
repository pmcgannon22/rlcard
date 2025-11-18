# Scout Training on M1 Mac - Complete Guide

This guide shows you how to train the improved Scout agent on your M1/M2 Mac with optimal performance.

## Prerequisites

### 1. Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python 3.9+ (Apple Silicon optimized)
```bash
# Use Homebrew Python (optimized for Apple Silicon)
brew install python@3.11

# Verify installation
python3 --version  # Should be 3.11.x
```

### 3. Create Virtual Environment
```bash
cd /path/to/rlcard

# Create virtual environment
python3 -m venv venv_m1

# Activate it
source venv_m1/bin/activate

# You should see (venv_m1) in your prompt
```

## Install Dependencies

### Install PyTorch with MPS Support

**IMPORTANT**: M1/M2 Macs need PyTorch with MPS (Metal Performance Shaders) support for GPU acceleration.

```bash
# Install PyTorch with MPS support (for M1/M2)
pip install torch torchvision torchaudio

# Verify MPS is available
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# Should print: MPS available: True
```

If MPS is not available, you may need a newer PyTorch:
```bash
pip install --upgrade torch torchvision torchaudio
```

### Install Other Dependencies

```bash
# Install numpy and other dependencies
pip install numpy matplotlib pandas

# Install RLCard in development mode
pip install -e .
```

## Quick Test

### Test the Improvements
```bash
python examples/test_scout_improvements.py
```

Expected output:
```
==============================================================
TEST 1: Dense Encoding
==============================================================
State shape: [[112]]
Observation size: 112 features
Expected reduction: 688 → 112
Reduction: 83.7%
✓ Dense encoding working correctly!

==============================================================
TEST 2: Reward Shaping
==============================================================
Number of shaped rewards collected:
  Player 0: 15 rewards
✓ Reward shaping working correctly!

==============================================================
TEST 3: Action Features
==============================================================
✓ Action features working correctly!

==============================================================
ALL TESTS PASSED! ✓
==============================================================
```

## Training Options

### Option 1: Quick Test Run (Recommended First)
```bash
python examples/train_scout_improved_m1.py \
    --num_episodes 1000 \
    --evaluate_every 100 \
    --log_dir experiments/scout_test/
```

This runs for 1000 episodes (~5-10 minutes on M1) to verify everything works.

### Option 2: Standard Training Run
```bash
python examples/train_scout_improved_m1.py \
    --num_episodes 10000 \
    --evaluate_every 200 \
    --log_dir experiments/scout_10k/
```

Expected time: 1-2 hours on M1 Pro/Max, 2-4 hours on M1 base.

### Option 3: Extended Training Run
```bash
python examples/train_scout_improved_m1.py \
    --num_episodes 20000 \
    --evaluate_every 500 \
    --mlp_layers 256 256 128 \
    --learning_rate 0.00005 \
    --log_dir experiments/scout_20k/
```

Expected time: 3-6 hours on M1 Pro/Max.

### Option 4: Advanced Configuration
```bash
python examples/train_scout_improved_m1.py \
    --num_episodes 30000 \
    --evaluate_every 500 \
    --mlp_layers 512 256 128 \
    --learning_rate 0.0001 \
    --epsilon_decay_steps 40000 \
    --batch_size 64 \
    --replay_memory_size 20000 \
    --save_every 2000 \
    --log_dir experiments/scout_advanced/
```

## M1-Specific Optimizations

The M1 training script includes several optimizations:

### 1. MPS GPU Acceleration
- Automatically uses Metal Performance Shaders
- 2-3x faster than CPU on M1
- 3-5x faster on M1 Pro/Max

### 2. Memory Management
- **Batch size**: 32 (optimal for M1 unified memory)
- **Replay memory**: 15,000 transitions (balanced for M1)
- Larger M1 Pro/Max can use bigger values:
  ```bash
  --batch_size 64 --replay_memory_size 25000
  ```

### 3. Evaluation Frequency
- More frequent evaluation (every 200 episodes)
- Smaller evaluation games (500 vs 1000)
- Faster feedback on training progress

## Monitoring Training

### Live Monitoring
```bash
# In a separate terminal, watch the log file
tail -f experiments/scout_10k/log.txt
```

### Check Progress
```bash
# View the CSV results
cat experiments/scout_10k/performance.csv

# Or with column formatting
column -t -s ',' experiments/scout_10k/performance.csv | less
```

### View Learning Curve
The script generates `performance.png` in the log directory. Open it to see the learning curve:
```bash
open experiments/scout_10k/performance.png
```

## Expected Performance

### Timeline (on M1 Pro)
- **Episode 0-2000**: Random play (~0.0 reward vs random)
- **Episode 2000-5000**: Learning basic strategies (+0.2 to +0.4 reward)
- **Episode 5000-10000**: Competent play (+0.5 to +0.7 reward)
- **Episode 10000+**: Strong play (+0.7 to +0.9 reward)

### Win Rates vs Random
- **Before improvements**: 60-70% after 50k episodes
- **With improvements**: 80-90% after 10k episodes

## Troubleshooting

### MPS Not Available
```bash
# Check PyTorch version (need 1.12+)
python3 -c "import torch; print(torch.__version__)"

# Update if needed
pip install --upgrade torch
```

### Memory Errors
If you get memory errors on M1 base (8GB):
```bash
python examples/train_scout_improved_m1.py \
    --batch_size 16 \
    --replay_memory_size 10000 \
    --num_episodes 10000
```

### Slow Training
If training is slower than expected:

1. **Check device**: Verify MPS is being used
   ```bash
   # Look for "MPS available" in output
   python examples/train_scout_improved_m1.py --num_episodes 10
   ```

2. **Close other apps**: Free up memory and GPU resources

3. **Reduce network size**: Use smaller layers
   ```bash
   --mlp_layers 128 128 64
   ```

### Training Crashes
If training crashes:

1. **Save more frequently**:
   ```bash
   --save_every 500
   ```

2. **Reduce batch size**:
   ```bash
   --batch_size 16
   ```

3. **Check disk space**: Logs and checkpoints can be large

## Resume Training

If training is interrupted, you can resume from a checkpoint:

```bash
python examples/train_scout_improved_m1.py \
    --load_checkpoint_path experiments/scout_10k/model.pth \
    --num_episodes 20000 \
    --log_dir experiments/scout_resumed/
```

## Performance Comparison

To compare improvements vs baseline:

```bash
# Train with improvements (new)
python examples/train_scout_improved_m1.py \
    --num_episodes 10000 \
    --log_dir experiments/with_improvements/

# Train without improvements (baseline)
python examples/run_rl.py \
    --env scout \
    --algorithm dqn \
    --num_episodes 10000 \
    --log_dir experiments/baseline/

# Compare results
echo "With improvements:"
tail -5 experiments/with_improvements/performance.csv
echo "Baseline:"
tail -5 experiments/baseline/performance.csv
```

## Hardware-Specific Recommendations

### M1 Base (8GB RAM)
```bash
--batch_size 16 \
--replay_memory_size 10000 \
--mlp_layers 128 128 64 \
--num_episodes 10000
```

### M1 Pro/Max (16GB+ RAM)
```bash
--batch_size 64 \
--replay_memory_size 25000 \
--mlp_layers 512 256 128 \
--num_episodes 20000
```

### M2 Pro/Max (32GB+ RAM)
```bash
--batch_size 128 \
--replay_memory_size 40000 \
--mlp_layers 512 512 256 \
--num_episodes 30000
```

## Background Training

For long training runs, use `nohup`:

```bash
nohup python examples/train_scout_improved_m1.py \
    --num_episodes 20000 \
    --log_dir experiments/scout_20k/ \
    > training.log 2>&1 &

# Check progress
tail -f training.log

# Or check the process
ps aux | grep train_scout
```

## Next Steps After Training

1. **Evaluate the trained agent**:
   ```bash
   python examples/human/scout_human.py \
       --checkpoint experiments/scout_10k/model.pth
   ```

2. **Analyze performance**:
   - Check `performance.csv` for reward over time
   - View `performance.png` for learning curve
   - Compare win rates vs random

3. **Further improvements**:
   - See `SCOUT_IMPROVEMENTS.md` for Phase 2-4 improvements
   - Implement self-play for stronger agents
   - Add curriculum learning

## Getting Help

If you encounter issues:

1. Check the test suite runs: `python examples/test_scout_improvements.py`
2. Verify MPS is available: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. Check the logs: `cat experiments/*/log.txt`
4. Review this guide's troubleshooting section

## Summary Commands

```bash
# Complete setup from scratch
python3 -m venv venv_m1
source venv_m1/bin/activate
pip install torch numpy matplotlib pandas
pip install -e .

# Test
python examples/test_scout_improvements.py

# Train (recommended for M1)
python examples/train_scout_improved_m1.py \
    --num_episodes 10000 \
    --log_dir experiments/scout_m1/

# Monitor
tail -f experiments/scout_m1/log.txt
```

Happy training! 🚀
