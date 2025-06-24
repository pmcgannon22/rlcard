# Scout Game Analysis Prompts

## Quick Analysis Prompt

```
You are analyzing Scout card game logs from AI agents. Scout is a strategic card game where players can PLAY cards (forming runs/groups) or SCOUT cards (taking from table). Cards have top/bottom values (e.g., "A/2").

Analyze the provided game logs to answer:

1. **Performance**: Win rates, average payoffs, game lengths
2. **Strategy**: Play vs scout ratios, card selection patterns, scouting preferences
3. **Agent Differences**: How DQN agents differ from Random agents
4. **Key Patterns**: Notable strategies, decision-making patterns, game flow insights

Focus on actionable insights about agent behavior and strategic decision-making.
```

## Detailed Strategic Analysis Prompt

```
You are an expert game theorist analyzing Scout card game agent behavior. 

**Game Context**: Scout is a strategic card game where players either PLAY cards (forming consecutive runs or same-value groups) or SCOUT cards (taking from table, optionally flipping to use bottom value). Players score by playing stronger sets than the table.

**Analysis Focus**:
- Strategic decision-making patterns
- Play vs Scout trade-offs
- Card selection and combination strategies
- Adaptation to game state and opponent behavior
- Learning evidence in trained agents vs random agents

**Specific Questions**:
1. What patterns emerge in when agents choose to play vs scout?
2. How do agents select which cards to play together?
3. What scouting strategies do agents employ (front vs back, flipping)?
4. How do agents adapt to different game states (empty table, few cards left)?
5. What distinguishes successful from unsuccessful strategies?

Provide detailed analysis with specific examples from the logs.
```

## Comparative Agent Analysis Prompt

```
Analyze Scout game logs comparing different agent types (DQN vs Random).

**Key Comparison Areas**:
1. **Decision Patterns**: Play vs scout frequency, card selection
2. **Strategic Sophistication**: Evidence of planning vs random choices
3. **Performance Metrics**: Win rates, scoring efficiency, game length
4. **Adaptation**: Response to opponent behavior and game state
5. **Learning Evidence**: How DQN behavior differs from random baseline

**Analysis Questions**:
- Does the DQN agent show consistent strategic patterns?
- What strategies has the DQN agent learned that random agents don't use?
- How does the DQN agent's decision-making evolve during games?
- What are the key factors that lead to DQN wins vs losses?

Provide quantitative analysis with specific examples from the game logs.
```

## Performance Optimization Prompt

```
You are analyzing Scout agent performance to identify optimization opportunities.

**Performance Analysis**:
1. **Efficiency Metrics**: Points per move, cards played vs scouted
2. **Strategic Gaps**: Missed opportunities, suboptimal decisions
3. **Pattern Recognition**: Successful vs unsuccessful strategies
4. **Improvement Areas**: Specific recommendations for agent enhancement

**Key Questions**:
- What strategies lead to the highest win rates?
- Where do agents make suboptimal decisions?
- What patterns emerge in winning vs losing games?
- How could the agent's strategy be improved?

Focus on actionable insights for improving agent performance.
```

## Game Flow Analysis Prompt

```
Analyze the temporal patterns in Scout game logs to understand game flow dynamics.

**Flow Analysis Areas**:
1. **Opening Strategies**: First few moves and their impact
2. **Mid-Game Patterns**: Decision-making during active gameplay
3. **End-Game Tactics**: Strategies when cards are running low
4. **State Transitions**: How agents respond to changing game conditions

**Specific Questions**:
- How do opening moves influence game outcomes?
- What patterns emerge in mid-game decision-making?
- How do agents handle end-game situations?
- What triggers strategic shifts during games?

Provide timeline-based analysis with specific game sequences.
```

## Custom Analysis Template

```
You are analyzing Scout card game logs for [SPECIFIC FOCUS AREA].

**Context**: Scout is a strategic card game with PLAY (form sets) and SCOUT (take cards) actions. Cards have top/bottom values.

**Analysis Focus**: [DESCRIBE SPECIFIC AREA OF INTEREST]

**Key Questions**:
1. [QUESTION 1]
2. [QUESTION 2] 
3. [QUESTION 3]
4. [QUESTION 4]

**Output Format**:
- Executive Summary
- Detailed Findings
- Specific Examples from Logs
- Recommendations/Insights

Analyze the provided game logs according to this framework.
```

## Usage Instructions

1. **Choose the appropriate prompt** based on your analysis goals
2. **Copy the prompt** and paste it into your LLM interface
3. **Add the game logs** after the prompt
4. **Specify any additional questions** you want answered
5. **Request specific output format** if needed

## Example Usage

```
[PASTE PROMPT HERE]

Here are the game logs to analyze:

[PASTE GAME LOGS HERE]

Please focus specifically on:
- How the DQN agent's scouting behavior differs from Random agents
- What patterns emerge in successful vs unsuccessful games
- Recommendations for improving agent performance
```
