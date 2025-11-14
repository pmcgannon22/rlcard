from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path
    ROOT = _Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from web.scout_web_backend.game_manager import GameConfig, ScoutWebGame
else:
    from .game_manager import GameConfig, ScoutWebGame


def create_app(config: GameConfig, static_dir: Optional[Path]) -> Flask:
    static_folder = str(static_dir) if static_dir and static_dir.exists() else None
    app = Flask(__name__, static_folder=static_folder, static_url_path='/')
    CORS(app)
    game = ScoutWebGame(config)

    @app.route('/api/state', methods=['GET'])
    def get_state():
        return jsonify(game.serialize_state())

    @app.route('/api/new-game', methods=['POST'])
    def new_game():
        game.reset_game()
        return jsonify(game.serialize_state())

    @app.route('/api/action', methods=['POST'])
    def take_action():
        payload = request.get_json(force=True) or {}
        action_id = payload.get('action_id')
        if action_id is None:
            return jsonify({'error': 'action_id is required'}), 400
        try:
            action_id = int(action_id)
            state = game.apply_human_action(action_id)
            return jsonify(state)
        except ValueError as err:
            return jsonify({'error': str(err)}), 400

    @app.route('/api/scout', methods=['POST'])
    def scout_action():
        payload = request.get_json(force=True) or {}
        direction = payload.get('direction')
        insertion_index = payload.get('insertion_index')
        flip = bool(payload.get('flip', False))
        if direction not in ('front', 'back'):
            return jsonify({'error': "direction must be 'front' or 'back'"}), 400
        if insertion_index is None:
            return jsonify({'error': 'insertion_index is required'}), 400
        try:
            insertion_index = int(insertion_index)
            state = game.apply_scout_choice(direction, insertion_index, flip)
            return jsonify(state)
        except ValueError as err:
            return jsonify({'error': str(err)}), 400

    if static_folder:
        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve_frontend(path):
            target = Path(static_folder) / path
            if target.exists() and target.is_file():
                return send_from_directory(static_folder, path)
            return send_from_directory(static_folder, 'index.html')

    return app


def main():
    parser = argparse.ArgumentParser(description="Scout web UI backend")
    parser.add_argument('--host', default='127.0.0.1', help='Host interface (default 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000, help='Port (default 8000)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to DMC checkpoint (model.tar)')
    parser.add_argument('--human-position', type=int, default=0, help='Seat index for human player')
    parser.add_argument('--device', default='cpu', help='Device for DMC agents (cpu or CUDA index)')
    parser.add_argument('--static-dir', type=str, default=None,
                        help='Optional path to pre-built frontend assets (defaults to web/scout-ui/dist)')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint_path and not checkpoint_path.exists():
        raise SystemExit(f'Checkpoint not found: {checkpoint_path}')

    default_static = Path(__file__).resolve().parent.parent / 'scout-ui' / 'dist'
    static_dir = Path(args.static_dir).resolve() if args.static_dir else default_static

    config = GameConfig(
        checkpoint=checkpoint_path,
        human_position=args.human_position,
        device=args.device,
    )
    app = create_app(config, static_dir)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
