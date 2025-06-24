#!/bin/bash

# Prepare Scout Game Logs for LLM Analysis

echo "🤖 Scout Game Analysis Preparation"
echo "=================================="

# Default values
LOG_FILE=${1:-"logs/agent_games_20250623_165552.log"}
ANALYSIS_TYPE=${2:-"comprehensive"}
MAX_GAMES=${3:-""}
OUTPUT_FILE=${4:-""}

# Check if log file exists
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo ""
    echo "Available log files:"
    ls -la logs/ 2>/dev/null || echo "No logs directory found"
    echo ""
    echo "Usage:"
    echo "  ./examples/scripts/prepare_analysis.sh [log_file] [analysis_type] [max_games] [output_file]"
    echo ""
    echo "Analysis types: comprehensive, performance, strategy, comparison"
    exit 1
fi

echo "Configuration:"
echo "  Log file: $LOG_FILE"
echo "  Analysis type: $ANALYSIS_TYPE"
if [ ! -z "$MAX_GAMES" ]; then
    echo "  Max games: $MAX_GAMES"
fi
if [ ! -z "$OUTPUT_FILE" ]; then
    echo "  Output file: $OUTPUT_FILE"
fi
echo ""

# Run the preparation script
if [ ! -z "$MAX_GAMES" ]; then
    python3 examples/scripts/prepare_llm_analysis.py \
        --log_file "$LOG_FILE" \
        --analysis_type "$ANALYSIS_TYPE" \
        --max_games "$MAX_GAMES"
else
    python3 examples/scripts/prepare_llm_analysis.py \
        --log_file "$LOG_FILE" \
        --analysis_type "$ANALYSIS_TYPE"
fi

if [ ! -z "$OUTPUT_FILE" ]; then
    # Rename the generated file
    GENERATED_FILE=$(ls analysis_*.txt | tail -1)
    mv "$GENERATED_FILE" "$OUTPUT_FILE"
    echo "✅ Analysis file saved as: $OUTPUT_FILE"
else
    GENERATED_FILE=$(ls analysis_*.txt | tail -1)
    echo "✅ Analysis file saved as: $GENERATED_FILE"
fi

echo ""
echo "📋 Analysis types available:"
echo "  comprehensive - Full analysis of all aspects"
echo "  performance   - Focus on win rates and efficiency"
echo "  strategy      - Focus on decision-making patterns"
echo "  comparison    - Focus on agent differences"
echo ""
echo "💡 Example usage:"
echo "  ./examples/scripts/prepare_analysis.sh logs/agent_games.log strategy 10"
echo "  ./examples/scripts/prepare_analysis.sh logs/agent_games.log performance 5 my_analysis.txt"
