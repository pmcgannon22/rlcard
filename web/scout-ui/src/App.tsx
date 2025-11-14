import { useCallback, useEffect, useState } from 'react';
import type { ActionOption, Card, ScoutState } from './types';

const fetchJson = async (url: string, options?: RequestInit) => {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || 'Request failed');
  }
  return response.json();
};

function CardRow({ title, cards }: { title: string; cards: Card[] }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <h2>{title}</h2>
        <span className="count">{cards.length} cards</span>
      </div>
      <div className="card-row">
        {cards.length === 0 && <span className="empty-note">Empty</span>}
        {cards.map((card, idx) => (
          <div className="card" key={`${card.label}-${idx}`}>
            <div className="card-value">{card.label}</div>
            {typeof card.position === 'number' && (
              <div className="card-index">#{card.position}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [state, setState] = useState<ScoutState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionMode, setActionMode] = useState<'play' | 'scout'>('play');
  const [selectedDirection, setSelectedDirection] = useState<'front' | 'back' | null>(null);
  const [scoutFlip, setScoutFlip] = useState(false);
  const [scoutInsertion, setScoutInsertion] = useState<number | null>(null);

  const refreshState = useCallback(async () => {
    try {
      setError(null);
      const data: ScoutState = await fetchJson('/api/state');
      setState(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load state');
    }
  }, []);

  useEffect(() => {
    refreshState();
  }, [refreshState]);

  useEffect(() => {
    if (!state) {
      return;
    }
    if (state.play_options.length === 0 && state.scout_info.canScout) {
      setActionMode('scout');
    } else if (!state.scout_info.canScout && state.play_options.length > 0) {
      setActionMode('play');
    }
    const arrowSlots =
      state.scout_info.arrows.map((a) => a.slot) ?? state.scout_info.insertionSlots;
    const slots = arrowSlots.length ? arrowSlots : state.scout_info.insertionSlots;
    if (slots.length === 1) {
      setScoutInsertion(slots[0]);
    } else if (!slots.includes(scoutInsertion ?? -1)) {
      setScoutInsertion(null);
    }
    setSelectedDirection(null);
    setScoutFlip(false);
  }, [state]);

  const startNewGame = async () => {
    try {
      setLoading(true);
      setError(null);
      const data: ScoutState = await fetchJson('/api/new-game', { method: 'POST' });
      setState(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to start new game');
    } finally {
      setLoading(false);
    }
  };

  const playAction = async (action: ActionOption) => {
    if (!state || !state.legal_actions_available || loading) {
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const data: ScoutState = await fetchJson('/api/action', {
        method: 'POST',
        body: JSON.stringify({ action_id: action.action_id }),
      });
      setState(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Action failed');
    } finally {
      setLoading(false);
    }
  };

  const submitScout = async () => {
    if (!state || loading || !state.scout_info.canScout) {
      return;
    }
    if (!selectedDirection) {
      setError('Select a card to scout.');
      return;
    }
    if (scoutInsertion === null) {
      setError('Select a position to insert the scouted card.');
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const data: ScoutState = await fetchJson('/api/scout', {
        method: 'POST',
        body: JSON.stringify({
          direction: selectedDirection,
          insertion_index: scoutInsertion,
          flip: scoutFlip,
        }),
      });
      setState(data);
      setSelectedDirection(null);
      setScoutFlip(false);
      const slots = data.scout_info.insertionSlots;
      setScoutInsertion(slots.length === 1 ? slots[0] : null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scout action failed');
    } finally {
      setLoading(false);
    }
  };

  const suggestedAction = state?.play_options.find((a) => a.isSuggestion);
  const suggestedScout = state?.scout_info.actions.find(
    (a) => a.action_id === state.suggested_action_id,
  );
  const scoutEndsRound =
    !!state &&
    state.table_owner !== null &&
    state.scout_info.canScout &&
    state.consecutive_scouts + 1 >= (state.num_players || state.scores.length) - 1;

  const describeSlot = (slot: number) => {
    if (!state) return `Slot ${slot}`;
    if (slot === 0) return 'Before first card';
    if (slot === state.hand.length) return 'After last card';
    return `Between ${slot - 1} & ${slot}`;
  };
  const getSlotLabel = (slot: number) =>
    state?.scout_info.arrows.find((a) => a.slot === slot)?.label || describeSlot(slot);

  return (
    <>
      <div className="app-container">
      <header>
        <h1>Scout Web Interface</h1>
        <div className="header-actions">
          <button className="secondary" onClick={startNewGame} disabled={loading}>
            New Game
          </button>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}
      {!state && <div className="loading">Loading game state...</div>}

      {state && (
        <>
          <section className="status-bar">
            <div>
              <strong>Status:</strong> {state.action_prompt}
            </div>
            <div className={`status-item ${scoutEndsRound ? 'warning' : ''}`}>
              <strong>Consecutive scouts:</strong> {state.consecutive_scouts}
              {scoutEndsRound && <span className="warning-text"> ⚠ Next scout ends round</span>}
            </div>
            <div>
              <strong>Table owner:</strong>{' '}
              {state.table_owner === null ? 'None' : `Player ${state.table_owner}`}
            </div>
            <div>
              <strong>Your seat:</strong> Player {state.human_position}
            </div>
          </section>

          <section className="boards">
            <CardRow title="Your Hand" cards={state.hand} />
            <CardRow title="Table Set" cards={state.table} />
          </section>

          <section className="actions-panel">
            <div className="action-mode-toggle">
              <button
                className={actionMode === 'play' ? 'active' : ''}
                onClick={() => setActionMode('play')}
                disabled={state.play_options.length === 0}
              >
                Play
              </button>
              <button
                className={actionMode === 'scout' ? 'active' : ''}
                onClick={() => setActionMode('scout')}
                disabled={!state.scout_info.canScout}
              >
                Scout
              </button>
            </div>
            {actionMode === 'play' && (
              <>
                {state.play_options.length === 0 && (
                  <p className="empty-note">No playable sets available.</p>
                )}
                <div className="action-grid">
                  {state.play_options.map((action) => (
                    <button
                      key={action.action_id}
                      className={`action-button ${action.isSuggestion ? 'suggested' : ''}`}
                      disabled={state.game_over || loading}
                      onClick={() => playAction(action)}
                    >
                      <div className="action-title">
                        {action.isSuggestion && <span className="star">⭐</span>}
                        {action.title}
                      </div>
                      <div className="action-desc">{action.description}</div>
                    </button>
                  ))}
                </div>
                {suggestedAction && (
                  <div className="suggestion-note">
                    ⭐ Suggested move: {suggestedAction.title} ({suggestedAction.description})
                  </div>
                )}
              </>
            )}
            {actionMode === 'scout' && (
              <div className="scout-controls">
                {!state.scout_info.canScout && (
                  <p className="empty-note">Scouting is not allowed right now.</p>
                )}
                {state.scout_info.canScout && (
                  <>
                    <div className="scout-targets">
                      {state.scout_info.targets.map((target) => {
                        const isSuggested =
                          suggestedScout && suggestedScout.direction === target.direction;
                        return (
                          <button
                            key={target.direction}
                            className={`scout-target ${
                              selectedDirection === target.direction ? 'selected' : ''
                            } ${isSuggested ? 'suggested' : ''}`}
                            onClick={() => {
                              setSelectedDirection(target.direction);
                              if (!target.allowFlip) setScoutFlip(false);
                            }}
                          >
                            <div className="target-title">
                              {target.direction.toUpperCase()}{' '}
                              {isSuggested && <span className="star">⭐</span>}
                            </div>
                            <div className="target-card">{target.card.label}</div>
                            {!target.allowFlip && (
                              <div className="target-note">Flip unavailable</div>
                            )}
                          </button>
                        );
                      })}
                    </div>
                    {selectedDirection && (
                      <div className="scout-flip">
                        <label>
                          <input
                            type="checkbox"
                            checked={scoutFlip}
                            onChange={(e) => setScoutFlip(e.target.checked)}
                            disabled={
                              !state.scout_info.targets.find(
                                (t) => t.direction === selectedDirection && t.allowFlip,
                              )
                            }
                          />
                          Flip before inserting
                        </label>
                      </div>
                    )}
                    {selectedDirection &&
                      suggestedScout &&
                      suggestedScout.direction === selectedDirection && (
                        <div className="suggestion-note">
                          ⭐ Advisor prefers {suggestedScout.flip ? 'flipping' : 'keeping orientation'} for this scout.
                        </div>
                      )}
                    <div className="insertion-slots">
                      {state.scout_info.insertionSlots.map((slot) => (
                        <div key={slot} className="insertion-wrapper">
                          <div className="hand-gap">
                            <div className="gap-card">
                              {slot === 0 ? 'Start' : state.hand[slot - 1]?.label || '—'}
                            </div>
                            <div className="gap-card next">
                              {slot >= state.hand.length
                                ? 'End'
                                : state.hand[slot]?.label || '—'}
                            </div>
                            <div
                              className={`insertion-arrow ${
                                scoutInsertion === slot ? 'selected' : ''
                              } ${
                                suggestedScout && suggestedScout.insertion === slot ? 'suggested' : ''
                              }`}
                              onClick={() => setScoutInsertion(slot)}
                            >
                              <span className="arrow">↓</span>
                              <span className="arrow-label">{getSlotLabel(slot)}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                    {suggestedScout && (
                      <div className="suggestion-note">
                        ⭐ Suggested scout: {suggestedScout.direction} {suggestedScout.flip ? '(flip)' : '(keep)'} at{' '}
                        {getSlotLabel(suggestedScout.insertion)}
                      </div>
                    )}
                    <button
                      className="primary"
                      onClick={submitScout}
                      disabled={
                        loading ||
                        !state.scout_info.canScout ||
                        !selectedDirection ||
                        scoutInsertion === null
                      }
                    >
                      Confirm Scout
                    </button>
                  </>
                )}
              </div>
            )}
          </section>

          <section className="history-and-scores">
            <div className="panel">
              <h2>Recent Actions</h2>
              {state.recent_actions.length === 0 && (
                <p className="empty-note">No actions yet.</p>
              )}
              <ul className="history-list">
                {state.recent_actions.map((entry, idx) => (
                  <li key={`${entry.player}-${idx}`}>
                    <strong>Player {entry.player}:</strong> {entry.label}
                  </li>
                ))}
              </ul>
            </div>
            <div className="panel">
              <h2>Scores</h2>
              <table>
                <thead>
                  <tr>
                    <th>Player</th>
                    <th>Tokens</th>
                    <th>Hand Size</th>
                    <th>Payoff</th>
                  </tr>
                </thead>
                <tbody>
                  {state.scores.map((score) => (
                    <tr key={score.player}>
                      <td>{score.player === state.human_position ? 'You' : `Player ${score.player}`}</td>
                      <td>{score.score}</td>
                      <td>{score.hand_size}</td>
                      <td>{score.payoff ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {state.game_over && (
            <section className="panel highlight">
              <h2>Round Complete</h2>
              <p>{state.winner_text || 'Game finished.'}</p>
            </section>
          )}
          </>
        )}
      </div>

      {state?.game_over && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>
              {state.payoffs && state.payoffs[state.human_position] === Math.max(...state.payoffs)
                ? '🎉 Congratulations!'
                : 'Round Over'}
            </h2>
            <p>{state.winner_text || 'Game finished.'}</p>
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Tokens</th>
                  <th>Payoff</th>
                </tr>
              </thead>
              <tbody>
                {state.scores.map((score) => (
                  <tr key={score.player}>
                    <td>{score.player === state.human_position ? 'You' : `Player ${score.player}`}</td>
                    <td>{score.score}</td>
                    <td>{score.payoff ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button className="primary" onClick={startNewGame} disabled={loading}>
              Play Again
            </button>
          </div>
        </div>
      )}
    </>
  );
}
