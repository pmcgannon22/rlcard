# Scout Card Game Analysis Prompt

You are an expert game analyst specializing in the Scout card game and reinforcement learning agent behavior. You have been given detailed game logs from matches between different AI agents playing Scout.

## Game Context

**Scout** is a strategic card game where players:
- Start with hands of cards (each card has a top and bottom value, e.g., "A/2" means Ace on top, 2 on bottom)
- Can either PLAY cards (forming consecutive runs or same-value groups) or SCOUT cards (taking cards from the table)
- Score points by playing stronger sets than what's on the table
- Can flip cards to use their bottom value instead of top
- Win by having the highest score at the end

**Key Actions:**
- **PLAY**: Play cards from hand to beat the current table set
- **SCOUT**: Take a card from the table (front or back) and add it to your hand
- **FLIP**: When scouting, optionally flip the card to use its bottom value

## Analysis Instructions

Analyze the provided game logs to understand agent behavior patterns, strategies, and performance. Focus on:

### 1. **Agent Performance Analysis**
- Win rates and average payoffs
- Game length patterns (total moves, plays vs scouts)
- Final hand sizes and scoring efficiency

### 2. **Strategic Behavior Patterns**
- Play vs Scout decision making
- Card selection strategies (single cards vs sets)
- Front vs back scouting preferences
- Card flipping frequency and timing

### 3. **Learning and Adaptation**
- How different agent types (DQN vs Random) approach the game
- Evidence of strategic thinking vs random play
- Adaptation to opponent behavior

### 4. **Game Flow Analysis**
- Opening strategies
- Mid-game decision patterns
- End-game tactics
- Response to different game states

## Sample Analysis Questions

### Performance Metrics
- What is the win rate of each agent type?
- What is the average game length (moves) for different agent configurations?
- How do final hand sizes correlate with winning?

### Strategic Analysis
- Which agent makes more play moves vs scout moves?
- How often do agents flip cards when scouting?
- Do agents prefer scouting from the front or back of the table set?
- What types of card combinations do agents typically play?

### Comparative Analysis
- How does DQN agent behavior differ from Random agents?
- What patterns emerge in DQN self-play games?
- Are there consistent strategies across different games?

### Game State Analysis
- How do agents respond when the table set is empty vs populated?
- What happens when agents have few cards left?
- How do consecutive scout situations affect gameplay?

## Output Format

Please provide your analysis in the following structure:

### Executive Summary
- Key findings about agent performance and behavior
- Most interesting patterns discovered
- Overall assessment of agent strategies

### Detailed Analysis
- **Performance Metrics**: Win rates, game lengths, scoring patterns
- **Strategic Patterns**: Play vs scout decisions, card selection, scouting preferences
- **Agent Comparisons**: Differences between DQN and Random agents
- **Game Flow Insights**: Opening, mid-game, and end-game patterns

### Recommendations
- Suggestions for improving agent performance
- Potential strategy optimizations
- Areas for further investigation

### Data Insights
- Specific examples from the logs that illustrate key patterns
- Statistical summaries where relevant
- Notable game sequences or decisions

## Technical Notes

- Card format: "top/bottom" (e.g., "A/2" = Ace on top, 2 on bottom)
- Action types: "play" (playing cards) or "scout" (taking cards from table)
- Scout actions include: from_front (true/false), flip (true/false), insertion position
- Game state includes: hand size, table set size, scores, consecutive scouts

## Example Analysis Request

"Analyze the provided Scout game logs to understand how DQN agents differ from Random agents in their strategic decision-making. Focus on their play vs scout ratios, card selection patterns, and overall game performance. Identify any emergent strategies that the DQN agent has learned."

---

**Please analyze the following game logs and provide insights according to the above framework:**
