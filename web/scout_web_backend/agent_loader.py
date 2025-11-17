"""Agent loading and initialization for Scout web game."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch

from rlcard.agents import RandomAgent, DQNAgent
from rlcard.agents.dmc_agent.model import DMCAgent, DMCModel


def _align_state_dict_inputs(agent: DMCAgent, state_dict: dict) -> dict:
    """Pad or crop the first FC layer if observation size changed.

    This handles backward compatibility when loading checkpoints trained
    with different observation sizes.

    Args:
        agent: The DMC agent with the current network architecture.
        state_dict: The state dict loaded from checkpoint.

    Returns:
        Aligned state dict with matching tensor dimensions.
    """
    weight_key = 'fc_layers.0.weight'
    bias_key = 'fc_layers.0.bias'

    if weight_key not in state_dict or bias_key not in state_dict:
        return state_dict

    target_weight = agent.net.fc_layers[0].weight
    target_bias = agent.net.fc_layers[0].bias
    weight = state_dict[weight_key]
    bias = state_dict[bias_key]
    tw_shape = target_weight.shape

    if weight.shape != tw_shape:
        # Align output dimension
        if weight.shape[0] != tw_shape[0]:
            if weight.shape[0] > tw_shape[0]:
                weight = weight[:tw_shape[0], :]
            else:
                pad = torch.zeros(tw_shape[0] - weight.shape[0], weight.shape[1], dtype=weight.dtype)
                weight = torch.cat([weight, pad], dim=0)

        # Align input dimension
        if weight.shape[1] != tw_shape[1]:
            if weight.shape[1] > tw_shape[1]:
                weight = weight[:, :tw_shape[1]]
            else:
                pad = torch.zeros(weight.shape[0], tw_shape[1] - weight.shape[1], dtype=weight.dtype)
                weight = torch.cat([weight, pad], dim=1)

    if bias.shape != target_bias.shape:
        if bias.shape[0] > target_bias.shape[0]:
            bias = bias[:target_bias.shape[0]]
        else:
            pad = torch.zeros(target_bias.shape[0] - bias.shape[0], dtype=bias.dtype)
            bias = torch.cat([bias, pad], dim=0)

    new_state = dict(state_dict)
    new_state[weight_key] = weight
    new_state[bias_key] = bias
    return new_state


def load_dmc_agents(env, checkpoint_path: Path, device: str) -> List[DMCAgent]:
    """Load DMC agents from a checkpoint file.

    Args:
        env: The RLCard environment.
        checkpoint_path: Path to the checkpoint file (.tar).
        device: Device string for PyTorch (e.g., 'cpu' or 'cuda:0').

    Returns:
        List of loaded DMC agents, one per player.

    Raises:
        ValueError: If the checkpoint is missing required data.
    """
    data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_shape = env.state_shape
    action_shape = env.action_shape

    if action_shape[0] is None:
        action_shape = [[env.num_actions] for _ in range(env.num_players)]

    model = DMCModel(
        state_shape,
        action_shape,
        exp_epsilon=0.0,
        device=device,
    )

    state_dicts = data.get('model_state_dict')
    if not state_dicts:
        raise ValueError("Checkpoint is missing 'model_state_dict'")

    for pid, state_dict in enumerate(state_dicts):
        agent = model.get_agent(pid)
        aligned = _align_state_dict_inputs(agent, state_dict)
        agent.load_state_dict(aligned)
        agent.set_device('cpu' if device == 'cpu' else f'cuda:{device}')

    model.eval()
    return model.get_agents()


def load_dqn_agent(env, checkpoint_path: Path, device: str) -> DQNAgent:
    """Load a single DQN agent from a checkpoint file.

    Args:
        env: The RLCard environment.
        checkpoint_path: Path to the DQN agent checkpoint (.pth).
        device: Device string for PyTorch (e.g., 'cpu' or 'mps').

    Returns:
        Loaded DQN agent.

    Raises:
        ValueError: If the checkpoint cannot be loaded.
    """
    try:
        agent = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Verify it's a DQN agent
        if not isinstance(agent, DQNAgent):
            raise ValueError(f"Expected DQNAgent, got {type(agent).__name__}")

        # Set device
        if hasattr(agent, 'set_device'):
            agent.set_device(device)

        # Set to eval mode
        if hasattr(agent, 'eval'):
            agent.eval()

        return agent
    except Exception as e:
        raise ValueError(f"Failed to load DQN agent: {e}")


def initialize_agents(env, checkpoint_path: Path | None, device: str) -> List:
    """Initialize game agents based on configuration.

    Detects checkpoint type (DMC or DQN) and loads appropriately.
    - DMC checkpoints: Dict with 'model_state_dict' for all players
    - DQN checkpoints: Single DQNAgent object (trained as player 0)

    Args:
        env: The RLCard environment.
        checkpoint_path: Optional path to checkpoint. If None, uses random agents.
        device: Device string for PyTorch.

    Returns:
        List of agents for each player position.
    """
    if not checkpoint_path:
        return [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]

    # Load checkpoint to detect type
    try:
        data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint {checkpoint_path}: {e}")

    # Detect checkpoint type
    if isinstance(data, DQNAgent):
        # DQN agent checkpoint - single agent trained as player 0
        print(f"Detected DQN agent checkpoint")
        agent = data

        # Set device
        if hasattr(agent, 'set_device'):
            agent.set_device(device)

        # Create agent list: DQN agent for all positions
        # (In multiplayer, all players use the same trained agent)
        agents = [agent for _ in range(env.num_players)]
        return agents

    elif isinstance(data, dict) and 'model_state_dict' in data:
        # DMC checkpoint - contains state dicts for all players
        print(f"Detected DMC checkpoint")
        checkpoint_path_obj = checkpoint_path if isinstance(checkpoint_path, Path) else Path(checkpoint_path)
        return load_dmc_agents(env, checkpoint_path_obj, device)

    else:
        raise ValueError(
            f"Unknown checkpoint format. Expected DQNAgent object or dict with 'model_state_dict', "
            f"got {type(data).__name__}"
        )
