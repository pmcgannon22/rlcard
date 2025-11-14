''' Play Scout against AI agents (random or a trained DMC checkpoint). '''
import argparse
import os
import torch

from rlcard.agents import RandomAgent
import rlcard
from rlcard.agents.dmc_agent.model import DMCModel
from rlcard.agents.human_agents.scout_human_agent import HumanAgent


def _resolve_action_shape(env):
    action_shape = env.action_shape
    if action_shape[0] is None:
        action_shape = [[env.num_actions] for _ in range(env.num_players)]
    return action_shape


def _load_checkpoint_agents(env, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dicts = checkpoint.get('model_state_dict')
    if not state_dicts:
        raise ValueError('Checkpoint missing "model_state_dict" entries.')

    model = DMCModel(
        env.state_shape,
        _resolve_action_shape(env),
        exp_epsilon=0.0,
        device=device,
    )
    for pid, state_dict in enumerate(state_dicts):
        model.get_agent(pid).load_state_dict(state_dict)
        # keep everything on CPU unless user explicitly requested cuda index
        model.get_agent(pid).set_device('cpu' if device == 'cpu' else f'cuda:{device}')
    model.eval()
    return model.get_agents()


def build_agent_list(env, human_pos, checkpoint, device):
    if checkpoint:
        ai_agents = _load_checkpoint_agents(env, checkpoint, device)
        advisor = ai_agents[human_pos]
    else:
        ai_agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
        advisor = None

    human_agent = HumanAgent(env.num_actions, advisor=advisor)

    ordered_agents = []
    for pid in range(env.num_players):
        if pid == human_pos:
            ordered_agents.append(human_agent)
        else:
            ordered_agents.append(ai_agents[pid])
    return ordered_agents, human_pos


def main():
    parser = argparse.ArgumentParser('Play Scout against AI agents')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to DMC checkpoint (model.tar). If omitted, opponents are random agents.')
    parser.add_argument('--human_position', type=int, default=0,
                        help='Seat index for the human player (0-based).')
    parser.add_argument('--device', default='cpu',
                        help='Device string for DMC agents ("cpu" or GPU index).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional environment seed.')
    args = parser.parse_args()

    config = {}
    if args.seed is not None:
        config['seed'] = args.seed
    env = rlcard.make('scout', config=config)

    if args.human_position < 0 or args.human_position >= env.num_players:
        raise ValueError(f'human_position must be within [0, {env.num_players-1}]')

    agents, human_pos = build_agent_list(env, args.human_position, args.checkpoint, args.device)
    env.set_agents(agents)

    print(">> Scout Game")
    print(f">> You are Player {human_pos}")
    if args.checkpoint:
        print(f">> Opponents use checkpoint: {args.checkpoint}")
    else:
        print(">> Opponents are random agents")
    print(">> Rules: Play sets of cards (runs or groups) or scout cards from the table")
    print(">> A set is stronger if it has more cards, or if same length, higher rank")
    print(">> If you can't play, you must scout (take a card from table)")
    print(">> First player to empty their hand wins!")
    print("")

    while True:
        print(">> Start a new game")
        print("=" * 60)

        trajectories, payoffs = env.run(is_training=False)
        
        final_state = trajectories[human_pos][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        
        _action_list = []
        for i in range(1, len(action_record)+1):
            if action_record[-i][0] == state.get('current_player', 0):
                break
            _action_list.insert(0, action_record[-i])
        
        if _action_list:
            print('\n=============== Final Actions ===============')
            for pair in _action_list:
                print('>> Player', pair[0], 'chooses ', end='')
                if hasattr(pair[1], 'get_action_repr'):
                    print(pair[1].get_action_repr())
                else:
                    print(pair[1])
            print('')

        print('=============== Final Game State ===============')
        perfect_info = env.get_perfect_information()
        for i in range(env.num_players):
            player_hand = perfect_info.get('hand_cards', [])[i] if i < len(perfect_info.get('hand_cards', [])) else []
            print(f'Player {i} hand: {player_hand}')
        print('')

        print('=============== Result ===============')
        payoff = payoffs[human_pos]
        max_payoff = max(payoffs)
        winners = [i for i, p in enumerate(payoffs) if p == max_payoff]
        if human_pos in winners:
            if len(winners) == 1:
                print('You win!')
            else:
                print('You tie for first place!')
        else:
            print('You lose.')
            print(f'Winner(s): {", ".join(str(i) for i in winners)} '
                  f'with payoff {max_payoff}')
        print(f'Your payoff: {payoff}')
        print('')
        
        print('=============== Final Scores ===============')
        for i in range(env.num_players):
            player_score = env.game.round.players[i].score
            player_hand_size = len(env.game.round.players[i].hand)
            player_type = "You" if i == human_pos else f"AI Player {i}"
            payoff_display = payoffs[i] if i < len(payoffs) else 'N/A'
            print(f'{player_type}: {player_score} tokens, payoff {payoff_display} '
                  f'(hand size: {player_hand_size})')
        print('')

        try:
            user_input = input("Press Enter to continue or 'q' to quit: ")
            if user_input.lower() == 'q':
                print("Thanks for playing!")
                break
        except KeyboardInterrupt:
            print("\nThanks for playing!")
            break


if __name__ == "__main__":
    main()
