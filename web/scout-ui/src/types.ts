export interface Card {
  top: number;
  bottom: number;
  label: string;
  position?: number;
}

export interface ActionOption {
  action_id: number;
  title: string;
  description: string;
  type: 'play' | 'scout' | 'other';
  isSuggestion: boolean;
}

export interface ActionLog {
  player: number;
  label: string;
  context: Record<string, unknown>;
}

export interface ScoreEntry {
  player: number;
  score: number;
  hand_size: number;
  payoff: number | null;
}

export interface ScoutTarget {
  direction: 'front' | 'back';
  card: Card;
  allowFlip: boolean;
}

export interface ScoutInfo {
  canScout: boolean;
  targets: ScoutTarget[];
  insertionSlots: number[];
  arrows: Array<{ slot: number; label: string }>;
  actions: Array<{ action_id: number; direction: 'front' | 'back'; flip: boolean; insertion: number }>;
}

export interface ScoutState {
  game_over: boolean;
  human_position: number;
  current_player: number;
  hand: Card[];
  table: Card[];
  legal_actions: ActionOption[];
  legal_actions_available: boolean;
  suggested_action_id?: number | null;
  play_options: ActionOption[];
  scout_info: ScoutInfo;
  recent_actions: ActionLog[];
  scores: ScoreEntry[];
  table_owner: number | null;
  consecutive_scouts: number;
  num_cards: Record<string, number>;
  payoffs: number[] | null;
  winner_text: string | null;
  action_prompt: string;
  num_players: number;
}
