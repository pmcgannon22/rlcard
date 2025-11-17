import { Fragment, useCallback, useEffect, useState } from 'react';
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

const TOP_COLORS = ['#fbbf24', '#f472b6', '#34d399', '#60a5fa', '#facc15'];
const BOTTOM_COLORS = ['#0f172a', '#1e293b', '#0f766e', '#78350f', '#312e81'];

const getPaletteColor = (value: number, palette: string[]) =>
  palette[(Math.max(1, value) - 1) % palette.length];

const getTopValue = (label?: string) =>
  (label && label.split('/')[0]) || label || '';

function ScoutCardView({ card }: { card: Card }) {
  const topColor = getPaletteColor(card.top, TOP_COLORS);
  const bottomColor = getPaletteColor(card.bottom, BOTTOM_COLORS);
  return (
    <div
      className="card"
      style={{
        background: `linear-gradient(180deg, ${topColor} 0%, ${topColor} 58%, ${bottomColor} 58%, ${bottomColor} 100%)`,
      }}
    >
      <div className="card-face">
        <div className="card-top">
          <span className="card-number">{card.top}</span>
          <span className="card-label">top</span>
        </div>
        <div className="card-bottom">
          <span className="card-number">{card.bottom}</span>
          <span className="card-label">bottom</span>
        </div>
      </div>
    </div>
  );
}

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
          <ScoutCardView card={card} key={`${card.label}-${idx}`} />
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

  const chooseOrientation = async (reverse: boolean) => {
    if (!state || loading) return;
    try {
      setLoading(true);
      setError(null);
      const data: ScoutState = await fetchJson('/api/orientation', {
        method: 'POST',
        body: JSON.stringify({ reverse }),
      });
      setState(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to set orientation');
    } finally {
      setLoading(false);
    }
  };

  const toggleAdvisor = async (enabled: boolean) => {
    try {
      setLoading(true);
      setError(null);
      const data: ScoutState = await fetchJson('/api/advisor', {
        method: 'POST',
        body: JSON.stringify({ enabled }),
      });
      setState(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update advisor setting');
    } finally {
      setLoading(false);
    }
  };

  const toggleDebug = async (enabled: boolean) => {
    try {
      setLoading(true);
      setError(null);
      const data: ScoutState = await fetchJson('/api/debug', {
        method: 'POST',
        body: JSON.stringify({ enabled }),
      });
      setState(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update debug setting');
    } finally {
      setLoading(false);
    }
  };

  const applySuggestedPlay = () => {
    if (suggestedAction) {
      playAction(suggestedAction);
    }
  };

  const applySuggestedScout = async () => {
    if (!state || !suggestedScout || loading) return;
    try {
      setLoading(true);
      setError(null);
      const data: ScoutState = await fetchJson('/api/scout', {
        method: 'POST',
        body: JSON.stringify({
          direction: suggestedScout.direction,
          insertion_index: suggestedScout.insertion,
          flip: suggestedScout.flip,
        }),
      });
      setState(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to apply scout suggestion');
    } finally {
      setLoading(false);
    }
  };

  const suggestedAction =
    state && state.advisor_enabled
      ? state.play_options.find((a) => a.isSuggestion)
      : undefined;
  const suggestedScout =
    state && state.advisor_enabled
      ? state.scout_info.actions.find((a) => a.action_id === state.suggested_action_id)
      : undefined;

  const formatValue = (value?: number | null, precision = 3) =>
    typeof value === 'number' ? value.toFixed(precision) : null;

  const scoutActionValue = () => {
    if (!state || !state.debug_enabled) return null;
    if (selectedDirection === null || scoutInsertion === null) return null;
    const candidates = state.scout_info.actions.filter(
      (a) =>
        a.direction === selectedDirection &&
        a.insertion === scoutInsertion &&
        Boolean(a.flip) === Boolean(scoutFlip),
    );
    if (candidates.length === 0) return null;
    const numeric = candidates.map((a) => a.value).filter((v): v is number => typeof v === 'number');
    if (!numeric.length) return null;
    return Math.max(...numeric);
  };

  const bestDirectionValue = (direction: 'front' | 'back') => {
    if (!state || !state.debug_enabled) return null;
    const candidates = state.scout_info.actions.filter((a) => a.direction === direction);
    if (candidates.length === 0) return null;
    const numeric = candidates.map((c) => c.value).filter((v): v is number => typeof v === 'number');
    if (numeric.length === 0) return null;
    return Math.max(...numeric);
  };

  const slotValue = (slot: number) => {
    if (!state || !state.debug_enabled) return null;
    const numeric = state.scout_info.actions
      .filter((a) => a.insertion === slot)
      .map((a) => a.value)
      .filter((v): v is number => typeof v === 'number');
    if (numeric.length === 0) return null;
    return Math.max(...numeric);
  };

  const renderArrowSlot = (slot: number) => {
    if (!state) return null;
    if (!state.scout_info.insertionSlots.includes(slot)) return null;
    const isSuggested =
      state.advisor_enabled && suggestedScout && suggestedScout.insertion === slot;
    const value = state.debug_enabled ? slotValue(slot) : null;
    return (
      <div
        className={`arrow-slot ${scoutInsertion === slot ? 'selected' : ''} ${
          isSuggested ? 'suggested' : ''
        }`}
        key={`slot-${slot}`}
      >
        <button onClick={() => setScoutInsertion(slot)} disabled={loading}>
          ↓
        </button>
        <span>
          {getSlotLabel(slot)}
          {state.debug_enabled && value !== null && (
            <span className="value-chip tiny">V {value.toFixed(3)}</span>
          )}
        </span>
        {state.advisor_enabled && suggestedScout && suggestedScout.insertion === slot && (
          <button className="outline" onClick={() => setScoutInsertion(slot)}>
            Use Suggested Slot
          </button>
        )}
      </div>
    );
  };

  const renderScoutHandPreview = () => {
    if (!state) return null;
    if (state.hand.length === 0) {
      return <div className="scout-hand-preview">{renderArrowSlot(0)}</div>;
    }
    return (
      <div className="scout-hand-preview">
        {state.hand.map((card, idx) => (
          <Fragment key={`preview-${idx}`}>
            {idx === 0 && renderArrowSlot(0)}
            <div className="mini-card large">{getTopValue(card.label)}</div>
            {renderArrowSlot(idx + 1)}
          </Fragment>
        ))}
      </div>
    );
  };
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
        {state?.orientation_pending && !state.game_over && (
          <div className="banner">
            <span role="img" aria-label="orientation">🔁</span>
            Choose your hand orientation before playing.
            {state.advisor_enabled && state.orientation_advice && (
              <span className="advice-text">
                Advisor suggests{' '}
                {state.orientation_advice.recommendation === 'keep'
                  ? 'keeping the current order'
                  : state.orientation_advice.recommendation === 'reverse'
                  ? 'reversing the order'
                  : 'either orientation'}
                {typeof state.orientation_advice.keep_value === 'number' &&
                  typeof state.orientation_advice.reverse_value === 'number' && (
                    <> (keep {state.orientation_advice.keep_value.toFixed(3)}, reverse {state.orientation_advice.reverse_value.toFixed(3)})</>
                  )}
              </span>
            )}
            <button className="primary" onClick={() => chooseOrientation(false)} disabled={loading}>
              Keep order
            </button>
            <button className="primary danger" onClick={() => chooseOrientation(true)} disabled={loading}>
              Reverse order
            </button>
          </div>
        )}
      <header>
        <h1>Scout Web Interface</h1>
        <div className="header-actions">
          <button className="secondary" onClick={startNewGame} disabled={loading}>
            New Game
          </button>
          {state && (
            <>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={state.advisor_enabled}
                  onChange={() => toggleAdvisor(!state.advisor_enabled)}
                  disabled={loading}
                />
                <span>Advisor {state.advisor_enabled ? 'On' : 'Off'}</span>
              </label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={state.debug_enabled}
                  onChange={() => toggleDebug(!state.debug_enabled)}
                  disabled={loading}
                />
                <span>Debug {state.debug_enabled ? 'On' : 'Off'}</span>
              </label>
            </>
          )}
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
                disabled={state.play_options.length === 0 || state.orientation_pending}
              >
                Play
              </button>
              <button
                className={actionMode === 'scout' ? 'active' : ''}
                onClick={() => setActionMode('scout')}
                disabled={!state.scout_info.canScout || state.orientation_pending}
              >
                Scout
              </button>
            </div>
            {state.orientation_pending && (
              <p className="empty-note">Choose your hand orientation to continue.</p>
            )}
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
                        {state.debug_enabled && typeof action.value === 'number' && (
                          <span className="value-chip">V {action.value.toFixed(3)}</span>
                        )}
                      </div>
                      <div className="action-desc">{action.description}</div>
                    </button>
                  ))}
                </div>
                {state.advisor_enabled && suggestedAction && (
                  <div className="suggestion-note">
                    ⭐ Suggested move: {suggestedAction.title} ({suggestedAction.description})
                    <button className="outline" onClick={applySuggestedPlay} disabled={loading}>
                      Apply Suggestion
                    </button>
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
                          state.advisor_enabled &&
                          suggestedScout &&
                          suggestedScout.direction === target.direction;
                        const directionValue =
                          state.debug_enabled && typeof target.value === 'number'
                            ? target.value
                            : state.debug_enabled
                            ? bestDirectionValue(target.direction)
                            : null;
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
                              {state.debug_enabled && directionValue !== null && (
                                <span className="value-chip tiny">V {directionValue.toFixed(3)}</span>
                              )}
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
                      state.advisor_enabled &&
                      suggestedScout &&
                      suggestedScout.direction === selectedDirection && (
                        <div className="suggestion-note">
                          ⭐ Advisor prefers {suggestedScout.flip ? 'flipping' : 'keeping orientation'} for this scout.
                        </div>
                      )}
                    {renderScoutHandPreview()}
                    {state.advisor_enabled && suggestedScout && (
                      <div className="suggestion-note">
                        ⭐ Suggested scout: {suggestedScout.direction} {suggestedScout.flip ? '(flip)' : '(keep)'} at{' '}
                        {getSlotLabel(suggestedScout.insertion)}
                        <button className="outline" onClick={applySuggestedScout} disabled={loading}>
                          Apply Suggestion
                        </button>
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
                    {state.debug_enabled && (
                      <div className="debug-note">
                        {(() => {
                          const value = scoutActionValue();
                          return value !== null ? (
                            <span>Estimated value: {value.toFixed(3)}</span>
                          ) : (
                            <span>Estimated value: —</span>
                          );
                        })()}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </section>

          <section className="history-and-scores">
            <div className="panel">
              <h2>Recent Actions</h2>
              {state.recent_actions.length === 0 ? (
                <p className="empty-note">No actions yet.</p>
              ) : (
                <ul className="history-list">
                  {state.recent_actions.map((entry, idx) => {
                    const ctx = entry.context as Record<string, any> | undefined;
                    const isPlay = ctx?.action_type === 'play';
                    const isScout = ctx?.action_type === 'scout';
                    const cards = (ctx?.cards as string[]) || [];
                    const scoutCard = ctx?.card as string | undefined;
                    const actionType = ctx?.action_type as string | undefined;
                    return (
                      <li key={`${entry.player}-${idx}`}>
                        <div className="history-entry">
                          <div className="history-entry-header">
                            <strong>Player {entry.player}</strong>
                            {actionType && (
                              <span className={`history-type ${actionType}`}>
                                {actionType.toUpperCase()}
                              </span>
                            )}
                          </div>
                          {isPlay && cards.length > 0 ? (
                            <div className="history-cards">
                              {cards.map((label: string, i: number) => (
                                <div className="mini-card" key={`${label}-${i}`}>
                                  {getTopValue(label)}
                                </div>
                              ))}
                            </div>
                          ) : isScout && scoutCard ? (
                            <div className="history-cards">
                              <div className="mini-card">{getTopValue(scoutCard)}</div>
                            </div>
                          ) : (
                            <span>{entry.label}</span>
                          )}
                          {state.debug_enabled && typeof entry.value === 'number' && (
                            <span className="value-chip tiny">V {entry.value.toFixed(3)}</span>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
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
                    {state.debug_enabled && <th>State Value</th>}
                  </tr>
                </thead>
                <tbody>
                  {state.scores.map((score) => (
                    <tr key={score.player}>
                      <td>{score.player === state.human_position ? 'You' : `Player ${score.player}`}</td>
                      <td>{score.score}</td>
                      <td>{score.hand_size}</td>
                      <td>{score.payoff ?? '—'}</td>
                      {state.debug_enabled && (
                        <td>{typeof score.state_value === 'number' ? score.state_value.toFixed(3) : '—'}</td>
                      )}
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
                  {state.debug_enabled && <th>State Value</th>}
                </tr>
              </thead>
              <tbody>
                {state.scores.map((score) => (
                  <tr key={score.player}>
                    <td>{score.player === state.human_position ? 'You' : `Player ${score.player}`}</td>
                    <td>{score.score}</td>
                    <td>{score.payoff ?? '—'}</td>
                    {state.debug_enabled && (
                      <td>{typeof score.state_value === 'number' ? score.state_value.toFixed(3) : '—'}</td>
                    )}
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
