# Scout Model Performance Analysis

## Problem Statement
After training for 10,000 episodes, the Scout model is missing "extremely obvious choices" in the Web UI. This document analyzes why and proposes solutions.

## Critical Issues Identified

### 1. **STATE ENCODING: Arbitrary Orientation Problem** ⚠️ HIGH IMPACT

**The Issue:**
Hand cards are encoded as `[top_value, bottom_value]`, but this creates a **representational ambiguity**.

**Example:**
```
Card A in hand: [top=5, bottom=3]  →  encoded as [0.5, 0.3]
Card B in hand: [top=3, bottom=5]  →  encoded as [0.3, 0.5]
```

**These cards are functionally IDENTICAL** because you can flip them when playing! But the network sees them as completely different states.

**Impact:**
- Network learns spurious patterns based on arbitrary initial orientation
- Doubles the effective state space (same game situation encoded differently)
- Confuses the network about which cards can form combos
- **This is likely the primary cause of "missing obvious choices"**

**Solution:**
Encode hand cards as `[min(top, bottom), max(top, bottom)]` to make representation **orientation-invariant**.

### 2. **STATE ENCODING: Table Card Bottom Values** ⚠️ MEDIUM IMPACT

**The Issue:**
Table cards encode BOTH top and bottom values:
```python
table_cards[j, 0] = card.rank      # visible top value ✓
table_cards[j, 1] = card.bottom    # HIDDEN bottom value ✗
```

The bottom value of table cards is **face-down and irrelevant** for gameplay decisions. You can only scout the visible top value.

**Impact:**
- Adds noise to the state representation
- Network may learn to rely on information it shouldn't have access to
- Wastes network capacity on irrelevant features

**Solution:**
For table cards, only encode the visible top value, set bottom to 0 or remove it entirely.

### 3. **TRAINING: Only Playing Against Random Agents** ⚠️ HIGH IMPACT

**The Issue:**
The model trains exclusively against `RandomAgent` opponents for all 10,000 episodes.

**Impact:**
- Never observes good Scout play
- Develops policies that exploit random play, not optimal play
- No curriculum learning - difficulty never increases
- **Explains why it beats random but makes poor strategic choices**

**Solution:**
Implement **self-play** or **opponent curriculum**:
- Start with random opponents (episodes 0-2000)
- Gradually introduce past versions of the learning agent (episodes 2000-6000)
- Eventually train primarily against strong opponents (episodes 6000+)

### 4. **TRAINING: Insufficient Episodes for Complexity** ⚠️ MEDIUM IMPACT

**The Issue:**
Scout has:
- Large action space (~1000-2000+ actions depending on hand)
- Complex combo mechanics (runs, groups, singles)
- Strategic depth (when to scout vs play, positioning)

10,000 episodes = ~50,000-100,000 game states (5 players, ~10 turns each)

**For comparison:**
- Simple games (Blackjack): 10,000 episodes sufficient
- Medium complexity (Leduc): 100,000+ episodes
- High complexity (No-Limit Hold'em): 1,000,000+ episodes

**Solution:**
Train for 50,000-100,000 episodes minimum.

### 5. **REWARD SHAPING: Suboptimal Signals** ⚠️ LOW-MEDIUM IMPACT

**Current reward shaping (scout.py:169-199):**
```python
# 1. Reward hand reduction
hand_reduction * 0.1

# 2. Bonus for long combos
if hand_reduction >= 4:
    reward += 0.3
```

**The Issue:**
This encourages playing cards ASAP, but sometimes **keeping cards is strategic**:
- Holding high cards to counter future table states
- Keeping cards that form potential future combos
- Strategic scouting to build better combinations

**Impact:**
Model learns to play cards too aggressively rather than strategically.

**Solution:**
Reduce or remove hand reduction rewards. Focus on:
- Points gained (already weighted 2.0x - good!)
- Avoiding forced scouts (already penalized - good!)
- Consider rewarding "playing when you're table owner" (captures points)

## Why the Model Misses Obvious Choices

Based on the above analysis, here's what's likely happening:

### Scenario: Obvious Combo Missed

**Game State:**
```
Hand: [3, 4, 5, 7, 8]  (could play 3-4-5 run)
Table: [6]
```

**What should happen:**
Play 3-4-5 (3-card run beats single 6)

**What the model might do:**
- **Orientation confusion**: If cards are encoded with different orientations, network doesn't recognize they form a run
- **Learned from random opponents**: Random agents don't punish bad combo choices, so model never learned combo value
- **Greedy hand reduction**: Model learned "play any cards = good" instead of "play GOOD combos = good"

### Scenario: Strategic Scout Missed

**Game State:**
```
Hand: [2, 2, 8, 9]
Table: [7-7-7] (opponent owns table)
You cannot beat 7-7-7 with any cards
```

**What should happen:**
Scout from table to build a better hand

**What the model might do:**
- **Try to play anyway** (forced scout penalty is only -0.2, might seem worth trying)
- **Random opponents don't teach this**: Random agents don't create situations requiring strategic scouting

## Recommended Improvements (Priority Order)

### PRIORITY 1: Fix State Encoding 🔥
**Estimated Impact: 30-50% improvement**

1. **Make hand cards orientation-invariant:**
   ```python
   # Instead of:
   hand_cards[i, 0] = card.rank
   hand_cards[i, 1] = card.bottom

   # Use:
   hand_cards[i, 0] = min(card.rank, card.bottom) / rank_count
   hand_cards[i, 1] = max(card.rank, card.bottom) / rank_count
   ```

2. **Remove table card bottom values:**
   ```python
   # Only encode visible information
   table_cards[j, 0] = card.rank / rank_count
   table_cards[j, 1] = 0.0  # or remove this dimension entirely
   ```

### PRIORITY 2: Implement Self-Play 🔥
**Estimated Impact: 40-60% improvement**

```python
# Add a curriculum schedule
if episode < 2000:
    opponents = [RandomAgent() for _ in range(4)]
elif episode < 6000:
    # Mix of random and past checkpoints
    opponents = [
        RandomAgent(),
        RandomAgent(),
        load_checkpoint(f'episode_{episode-1000}.pt'),
        load_checkpoint(f'episode_{episode-2000}.pt')
    ]
else:
    # Mostly self-play
    opponents = [
        RandomAgent(),  # 1 random for diversity
        load_checkpoint(f'episode_{episode-1000}.pt'),
        load_checkpoint(f'episode_{episode-2000}.pt'),
        load_checkpoint(f'episode_{episode-3000}.pt'),
    ]
```

### PRIORITY 3: Train Longer
**Estimated Impact: 20-30% improvement**

- Increase to 50,000-100,000 episodes
- Save checkpoints every 5,000 episodes
- Use early ones for opponent pool

### PRIORITY 4: Refine Reward Shaping
**Estimated Impact: 10-20% improvement**

```python
def _compute_shaped_reward(self, prev_state, action, next_state):
    reward = 0.0

    # Primary signal: points gained (KEEP THIS - it's good)
    points_gained = next_state.get('points', 0) - prev_state.get('points', 0)
    reward += points_gained * 2.0

    # Penalty for forced scout (KEEP THIS)
    if next_state.get('current_player_forced_scout', False):
        reward -= 0.2

    # REMOVE hand reduction rewards - they encourage greedy play
    # REMOVE combo bonuses - let points gained be the signal

    # ADD: Bonus for capturing table ownership (strategic)
    if (next_state.get('table_owner') == player_id and
        prev_state.get('table_owner') != player_id):
        reward += 0.3

    return reward
```

### PRIORITY 5: Add Combo-Aware Features
**Estimated Impact: 10-15% improvement**

Add features that help the network understand hand composition:
- Number of pairs in hand
- Number of possible runs in hand
- Longest run available
- Longest group available

## Next Steps

Would you like me to:
1. **Implement Priority 1** (fix state encoding) - Quick win, likely big impact
2. **Implement Priority 2** (self-play) - More complex, very high impact
3. **Both** - Maximum improvement potential

After fixing the encoding, you'll need to **retrain from scratch** because the state representation will change.

## Expected Results

With Priority 1 + 2 + longer training:
- Model should recognize obvious combos consistently
- Strategic scouting decisions improve dramatically
- Win rate vs random should reach 80-90% (currently unknown, but likely lower)
- Play quality visible in Web UI should appear much more "intelligent"

## Technical Note

The orientation invariance fix is similar to successful techniques in other card game RL:
- **Poker**: Suit abstraction (♠A-♠K ≡ ♥A-♥K for strategy)
- **Bridge**: Card equivalence within suits
- **Hanabi**: Encoding possible values rather than known values

Scout's flippable cards are analogous - we should encode the **possibility space** (what values can this card represent) rather than the **arbitrary current state** (which way is it currently facing in hand).
