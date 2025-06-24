#!/bin/bash

# Scout Agent Game Logger - Complete Version
# Logs agent gameplay for LLM analysis with detailed action decoding

echo "📊 Scout Agent Game Logger - Complete Version"
echo "============================================="

# Default values
NUM_GAMES=${1:-10}
CONFIG=${2:-"dqn_vs_random"}
LOG_FILE=${3:-"logs/agent_games_$(date +%Y%m%d_%H%M%S).log"}

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Configuration:"
echo "  Games to log: $NUM_GAMES"
echo "  Agent config: $CONFIG"
echo "  Log file: $LOG_FILE"
echo ""

# Run the logging script
python3 examples/scripts/log_agent_games.py \
    --num_games $NUM_GAMES \
    --agents_config $CONFIG \
    --log_file $LOG_FILE

echo ""
echo "✅ Logging complete!"
echo "📁 Log file: $LOG_FILE"
echo ""
echo "📋 Log format includes:"
echo "  • Detailed action descriptions (play/scout with card details)"
echo "  • Game state before each move"
echo "  • Final game state and payoffs"
echo "  • Game summary with move statistics"
echo ""
echo "🤖 Available agent configurations:"
echo "  dqn_vs_random    - DQN vs 3 Random agents"
echo "  dqn_vs_dqn       - 4 DQN agents (self-play)"
echo "  random_vs_random - 4 Random agents"
echo ""
echo "💡 Example usage:"
echo "  ./examples/scripts/log_games_complete.sh 50 dqn_vs_random"
echo "  ./examples/scripts/log_games_complete.sh 100 dqn_vs_dqn logs/dqn_selfplay.log"
