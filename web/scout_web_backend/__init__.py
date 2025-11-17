"""Scout web backend package.

This package provides a web-based interface for playing the Scout card game
with human and AI players.
"""

from __future__ import annotations

from .config import GameConfig
from .game import ScoutWebGame

__all__ = ['GameConfig', 'ScoutWebGame']
