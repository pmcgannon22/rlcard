# Improved State Encoding for Scout

## Summary of Changes

This document describes the state encoding improvements implemented to address the model's inability to recognize obvious plays.

## Problem

The original dense encoding had critical issues:

1. **Orientation Ambiguity**: Hand cards encoded as `[top, bottom]` treated functionally identical cards as different states
2. **Irrelevant Information**: Table cards encoded hidden bottom values
3. **No Self-Play**: Training only against random agents

## Solution

### 1. Improved State Encoding

#### Hand Cards (Before vs After)

**Before (112 features total):**
```python
# Each hand card: [top_value, bottom_value]
hand_cards = np.zeros((hand_size, 2))  # 9 * 2 = 18 features
table_cards = np.zeros((hand_size, 2))  # 9 * 2 = 18 features
```

**After (31 features total):**
```python
# Hand cards: only top value (what you play)
hand_cards = np.zeros(hand_size)  # 9 features

# Table cards: scoutable positions with both orientations
table_cards = np.zeros(4)  # [front_top, front_bottom, back_top, back_bottom]
```

#### Rationale

**Hand Cards - Top Only:**
- You cannot flip cards in your own hand
- The top value is what matters when playing
- Bottom value in hand adds no decision-making value
- Eliminates orientation ambiguity

**Table Cards - 4 Values:**
- You can only scout from front or back of table
- When scouting, you can choose to flip the card
- Encode both orientations of scoutable cards: `[front_top, front_bottom, back_top, back_bottom]`
- Provides all information needed for scouting decisions
- Removes middle cards (not scoutable, irrelevant)

### 2. State Size Reduction

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| Hand cards | 18 | 9 | -50% |
| Table cards | 18 | 4 | -78% |
| Hand mask | 9 | 9 | 0% |
| Table mask | 9 | 0 | -100% |
| Info vector | 18 | 16 | -11% |
| **TOTAL** | **72** | **38** | **-47%** |

**Breakdown:**
- Hand cards: 9 (one value per position)
- Table cards: 4 (front_top, front_bottom, back_top, back_bottom)
- Hand mask: 9 (which positions occupied)
- Info vector: 16
  - Owner one-hot: 6 (5 players + none)
  - Consecutive scouts: 1
  - Num cards per player: 5
  - Points: 1
  - Table size: 1
  - Orientation: 1
  - Forced scout: 1

### 3. Self-Play Curriculum

**Phase 1: Random Opponents (Episodes 0-2000)**
- All opponents are RandomAgent
- Builds basic understanding of game mechanics
- Learns legal plays and basic combos

**Phase 2: Mixed Training (Episodes 2000-6000)**
- 50% random opponents
- 50% past checkpoints (from checkpoint pool)
- Starts experiencing better opponents
- Learns to counter non-random play

**Phase 3: Self-Play (Episodes 6000+)**
- 1 random opponent (for diversity)
- 3 past checkpoints (strong opponents)
- Learns advanced strategy
- Improves against skilled play

**Checkpoint Pool Management:**
- Saves agent snapshot every 500 episodes
- Maintains pool of 10 most recent checkpoints
- Opponents randomly sampled from pool
- Ensures diversity in opponent strength

## Expected Improvements

### 1. Better Combo Recognition (High Impact)

**Before:** Network confused by orientation
```
Hand: [top=3,bottom=5], [top=4,bottom=6], [top=5,bottom=7]
Encoded as: [0.3, 0.5, 0.4, 0.6, 0.5, 0.7]
Network struggles to see this is a run (3-4-5)
```

**After:** Clear representation
```
Hand: [3, 4, 5, ...]
Encoded as: [0.3, 0.4, 0.5, ...]
Network easily recognizes consecutive values = run
```

### 2. Better Scouting Decisions (High Impact)

**Before:** Entire table encoded, bottom values create noise
```
Table: [1↕2, 3↕4, 5↕6, 7↕8, 9↕10]
Encoded all 10 cards with top+bottom
Most cards not scoutable (only front/back)
```

**After:** Only relevant information
```
Table: [...many cards...]
Encoded: [front_top=1, front_bottom=2, back_top=9, back_bottom=10]
Network knows exactly what scouting options are available
Can evaluate both flip orientations
```

### 3. Strategic Play from Self-Play (Very High Impact)

**Before:**
- Model only saw random play patterns
- Learned to exploit randomness
- Never punished for suboptimal choices
- No strategic depth

**After:**
- Model experiences progressively stronger opponents
- Learns what actually wins games
- Discovers advanced strategies through self-play
- Continuous improvement from facing better versions of itself

## Implementation Details

### State Encoding (`rlcard/envs/scout.py`)

```python
def _build_observation_vector(self, raw_state: dict) -> np.ndarray:
    # Hand cards: only top value
    hand_cards = np.zeros(self.hand_size, dtype=np.float32)
    for i, card in enumerate(raw_state['hand']):
        hand_cards[i] = self._normalize_rank_value(card.rank)

    # Table cards: front and back with both orientations
    table_cards = np.zeros(4, dtype=np.float32)
    if len(table_set) > 0:
        front_card = table_set[0]
        table_cards[0] = self._normalize_rank_value(front_card.rank)      # front top
        table_cards[1] = self._normalize_rank_value(front_card.bottom)    # front bottom

        back_card = table_set[-1]
        table_cards[2] = self._normalize_rank_value(back_card.rank)       # back top
        table_cards[3] = self._normalize_rank_value(back_card.bottom)     # back bottom

    # Info vector (unchanged)
    # ... game state features ...

    return np.concatenate([hand_cards, table_cards, hand_mask, info_vec])
```

### Self-Play Curriculum (`examples/train_scout_improved_m1.py`)

```python
def get_opponents_for_episode(episode, num_opponents):
    if episode < args.curriculum_random_episodes:
        # Phase 1: All random
        return [RandomAgent() for _ in range(num_opponents)]

    elif episode < args.curriculum_mixed_episodes:
        # Phase 2: 50/50 mix
        num_random = num_opponents // 2
        num_past = num_opponents - num_random
        opponents = [RandomAgent() for _ in range(num_random)]
        opponents += [sample_from_checkpoint_pool() for _ in range(num_past)]
        return opponents

    else:
        # Phase 3: Mostly self-play
        opponents = [RandomAgent()]  # 1 random for diversity
        opponents += [sample_from_checkpoint_pool() for _ in range(num_opponents - 1)]
        return opponents

# Checkpoint pool management
if episode % args.checkpoint_pool_interval == 0:
    agent_copy = deepcopy(agent)
    checkpoint_pool.append((episode, agent_copy))
    if len(checkpoint_pool) > args.max_checkpoint_pool_size:
        checkpoint_pool.pop(0)
```

## Breaking Changes

### Models Must Be Retrained

**State shape changed from 72 → 38 features**

Old checkpoints are **incompatible** with the new encoding. You must:

1. Delete old checkpoints
2. Retrain from scratch with new encoding
3. Expect different (better) behavior

### Web UI Compatibility

The Web UI will automatically work with new checkpoints, but:
- Old checkpoints will fail to load (state shape mismatch)
- Delete contents of `experiments/scout_improved_m1/`
- Retrain fresh model

## Training Recommendations

### Recommended Settings (M1 Mac)

```bash
python examples/train_scout_improved_m1.py \
    --num_episodes 50000 \
    --curriculum_random_episodes 2000 \
    --curriculum_mixed_episodes 6000 \
    --checkpoint_pool_interval 500 \
    --max_checkpoint_pool_size 10 \
    --epsilon_decay_steps 30000
```

### Why 50,000 Episodes?

- Phase 1 (Random): 2,000 episodes - learn basics
- Phase 2 (Mixed): 4,000 episodes - learn strategy
- Phase 3 (Self-Play): 44,000 episodes - master advanced play

Scout is complex enough to benefit from extended self-play training.

### Expected Training Time (M1 Mac)

- ~5-10 seconds per episode (depending on game length)
- 50,000 episodes ≈ 70-140 hours (3-6 days)
- Recommend running overnight/over weekend
- Use `monitor_training.py` to track progress

## Validation

### How to Verify Improvements

1. **Combo Recognition Test:**
   - Start Web UI with trained model
   - Create hand with obvious run (e.g., 3-4-5-6)
   - Model should prioritize playing the run
   - Compare with old model (likely missed it)

2. **Scouting Decision Test:**
   - Create situation where scouting is clearly better than playing
   - Model should choose scout action
   - Old model might force-play and fail

3. **Win Rate Test:**
   ```python
   from rlcard.utils import evaluate_win_rate
   results = evaluate_win_rate(env, num_games=1000, player_id=0)
   print(f"Win rate: {results['win_rate']:.1f}%")
   ```
   - Expect 70-85% win rate vs random after 50k episodes
   - Old model: likely 50-60%

4. **Self-Play Progress:**
   - Monitor win rate during training
   - Should see initial improvement (phase 1-2)
   - Temporary plateau or dip when self-play starts (phase 3)
   - Continued improvement as agent learns from itself

## Files Modified

- `rlcard/envs/scout.py`: State encoding changes
- `examples/train_scout_improved_m1.py`: Self-play curriculum
- `SCOUT_MODEL_ANALYSIS.md`: Original problem analysis
- `IMPROVED_STATE_ENCODING.md`: This document

## Next Steps

1. **Delete old checkpoints:** `rm -rf experiments/scout_improved_m1/`
2. **Start training:** `python examples/train_scout_improved_m1.py --num_episodes 50000`
3. **Monitor progress:** `python examples/monitor_training.py experiments/scout_improved_m1/`
4. **Test in Web UI:** Load checkpoint after 10k+ episodes
5. **Compare with baseline:** Test against random opponents

## Troubleshooting

### "State shape mismatch" Error

You're trying to load an old checkpoint. Delete it and retrain.

### "MPS not available" Warning

Your PyTorch doesn't have MPS support. Install PyTorch 2.0+ or it will use CPU (slower but works).

### Win Rate Not Improving

- Check that self-play is activating (see console output)
- Verify checkpoint pool is being populated
- May need more episodes (try 100k for very strong play)
- Check epsilon is decaying (should reach 0.1 by episode 30k with default settings)

### Memory Issues on M1

Reduce batch size or replay memory:
```bash
--batch_size 16 --replay_memory_size 10000
```

## References

- Original analysis: `SCOUT_MODEL_ANALYSIS.md`
- AlphaGo: Used self-play to reach superhuman level
- OpenAI Five: Curriculum learning with increasingly strong opponents
- RLCard documentation: https://rlcard.org/
