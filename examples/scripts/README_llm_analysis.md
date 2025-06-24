# LLM Analysis System for Scout Game Logs

This directory contains a complete system for logging Scout agent gameplay and preparing it for LLM analysis.

## Quick Start

### 1. Generate Game Logs
```bash
# Log 50 games of DQN vs Random agents
./examples/scripts/log_games_complete.sh 50 dqn_vs_random

# Log 100 games of DQN self-play
./examples/scripts/log_games_complete.sh 100 dqn_vs_dqn
```

### 2. Prepare for LLM Analysis
```bash
# Create comprehensive analysis file
./examples/scripts/prepare_analysis.sh logs/agent_games_*.log comprehensive

# Create strategy-focused analysis (first 10 games)
./examples/scripts/prepare_analysis.sh logs/agent_games_*.log strategy 10

# Create performance analysis with custom output
./examples/scripts/prepare_analysis.sh logs/agent_games_*.log performance 5 my_analysis.txt
```

### 3. Use with LLM
Copy the generated analysis file content and paste it into your preferred LLM interface (ChatGPT, Claude, etc.).

## Files Overview

### Logging System
- `log_agent_games_complete.py` - Main logging script with action decoding
- `log_games_complete.sh` - Easy-to-use shell script for logging
- `README_logging.md` - Detailed logging documentation

### Analysis Preparation
- `prepare_llm_analysis.py` - Script to prepare logs for LLM analysis
- `prepare_analysis.sh` - Shell script wrapper for analysis preparation
- `llm_prompt_template.txt` - Base prompt template
- `analysis_prompts.md` - Collection of specialized prompts

## Analysis Types

### Comprehensive Analysis
```bash
./examples/scripts/prepare_analysis.sh logs/agent_games.log comprehensive
```
**Focus**: Complete analysis of all aspects including performance, strategy, and agent comparisons.

### Performance Analysis
```bash
./examples/scripts/prepare_analysis.sh logs/agent_games.log performance
```
**Focus**: Win rates, scoring efficiency, game length optimization, and performance metrics.

### Strategy Analysis
```bash
./examples/scripts/prepare_analysis.sh logs/agent_games.log strategy
```
**Focus**: Decision-making patterns, play vs scout ratios, card selection strategies, and scouting preferences.

### Comparison Analysis
```bash
./examples/scripts/prepare_analysis.sh logs/agent_games.log comparison
```
**Focus**: Differences between DQN and Random agents, learning evidence, and behavioral patterns.

## Example Workflow

### Step 1: Generate Logs
```bash
# Generate 100 games for analysis
./examples/scripts/log_games_complete.sh 100 dqn_vs_random
```

### Step 2: Prepare Analysis
```bash
# Create strategy-focused analysis
./examples/scripts/prepare_analysis.sh logs/agent_games_*.log strategy 20
```

### Step 3: LLM Analysis
1. Open the generated analysis file (e.g., `analysis_strategy_*.txt`)
2. Copy the entire content
3. Paste into your LLM interface
4. Ask follow-up questions based on the analysis

## Sample LLM Questions

After receiving the initial analysis, you can ask:

### Performance Questions
- "What's the optimal play vs scout ratio for winning?"
- "How do game length and win rate correlate?"
- "What are the key factors that lead to high scores?"

### Strategic Questions
- "When should agents prefer scouting from front vs back?"
- "What patterns emerge in successful card combinations?"
- "How do agents adapt their strategy based on hand size?"

### Comparative Questions
- "What specific strategies has the DQN agent learned?"
- "How does DQN decision-making differ from random agents?"
- "What are the most effective strategies against random opponents?"

### Optimization Questions
- "What improvements would you suggest for the DQN agent?"
- "Which strategies are most effective for winning?"
- "How could the agent's decision-making be enhanced?"

## Output Format

The LLM will provide analysis in this structure:

### Executive Summary
- Key findings and patterns
- Overall assessment of agent strategies

### Detailed Analysis
- Performance metrics and statistics
- Strategic behavior patterns
- Agent comparisons and differences
- Game flow insights

### Recommendations
- Performance improvement suggestions
- Strategy optimization ideas
- Areas for further investigation

### Data Insights
- Specific examples from the logs
- Statistical summaries
- Notable game sequences

## Advanced Usage

### Custom Analysis
```bash
# Create analysis with specific focus
python3 examples/scripts/prepare_llm_analysis.py \
    --log_file logs/agent_games.log \
    --analysis_type strategy \
    --max_games 15 \
    --output_file custom_analysis.txt
```

### Multiple Analysis Types
```bash
# Generate different analysis perspectives
./examples/scripts/prepare_analysis.sh logs/agent_games.log performance 10 perf_analysis.txt
./examples/scripts/prepare_analysis.sh logs/agent_games.log strategy 10 strat_analysis.txt
./examples/scripts/prepare_analysis.sh logs/agent_games.log comparison 10 comp_analysis.txt
```

### Batch Processing
```bash
# Process multiple log files
for log_file in logs/agent_games_*.log; do
    ./examples/scripts/prepare_analysis.sh "$log_file" comprehensive
done
```

## Tips for Better Analysis

1. **Use Multiple Game Sets**: Generate logs with different agent configurations
2. **Vary Game Counts**: Analyze both small (5-10) and large (50-100) game sets
3. **Ask Follow-up Questions**: Use the initial analysis to guide deeper investigation
4. **Compare Configurations**: Analyze DQN vs Random, DQN self-play, and Random vs Random
5. **Focus on Patterns**: Look for consistent strategies and decision-making patterns

## Troubleshooting

### Common Issues
- **No log files found**: Run the logging script first
- **Empty analysis**: Check that log files contain valid game data
- **Large files**: Use `--max_games` to limit the number of games analyzed

### File Locations
- Log files: `logs/agent_games_*.log`
- Analysis files: `analysis_*.txt` (in current directory)
- Scripts: `examples/scripts/`

This system provides a complete pipeline from game logging to LLM analysis, enabling deep insights into Scout agent behavior and strategies.
