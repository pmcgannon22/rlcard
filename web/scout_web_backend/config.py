"""Configuration for Scout web game."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class GameConfig:
    """Configuration for the Scout web game.

    Attributes:
        checkpoint: Optional path to a DMC model checkpoint file.
        human_position: The player position (0-indexed) for the human player.
        device: Device string for PyTorch (e.g., 'cpu' or 'cuda:0').
        advisor_enabled: Whether to enable AI advisor suggestions.
        debug_enabled: Whether to show debug information like action values.
    """
    checkpoint: Optional[Path] = None
    human_position: int = 0
    device: str = 'cpu'
    advisor_enabled: bool = True
    debug_enabled: bool = False
