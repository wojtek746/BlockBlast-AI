from mcts_ai import MCTS
from GameSimulator import GameSimulator
import numpy as np
import time


def test_mcts_performance():
    configurations = [
        {"time_limit": 0.5, "c_param": 1.4, "name": "Fast (0.5s)"},
        {"time_limit": 1.0, "c_param": 1.4, "name": "Medium (1.0s)"},
        {"time_limit": 2.0, "c_param": 1.4, "name": "Slow (2.0s)"},
        {"time_limit": 1.0, "c_param": 0.7, "name": "Low exploration"},
        {"time_limit": 1.0, "c_param": 2.1, "name": "High exploration"},
    ]

    results = {}

    for config in configurations:
        print(f"\n{'=' * 50}")
        print(f"Testing: {config['name']}")
        print(f"Time limit: {config['time_limit']}s, C_param: {config['c_param']}")

        scores = []
        total_time = 0
        num_games = 10  # Zmień na więcej dla lepszych statystyk

        for game_num in range(num_games):
            print(f"Game {game_num + 1}/{num_games}...", end=" ")

            game = GameSimulator()
            game.start(game_num)

            mcts = MCTS(c_param=config['c_param'], time_limit=config['time_limit'])

            game_start = time.time()
            score = play_single_game(game, mcts, verbose=False)
            game_time = time.time() - game_start

            scores.append(score)
            total_time += game_time

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = max(scores)
        min_score = min(scores)
        avg_time = total_time / num_games

        results[config['name']] = {
            'scores': scores,
            'avg': avg_score,
            'std': std_score,
            'max': max_score,
            'min': min_score,
            'avg_time': avg_time
        }

        print(f"Results: Avg={avg_score:.1f}, Std={std_score:.1f}, Max={max_score}, Min={min_score}, Time={avg_time:.1f}s")

    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    print(f"{'Configuration':<20} {'Avg Score':<10} {'Max Score':<10} {'Avg Time':<10}")
    print("-" * 60)

    for name, result in results.items():
        print(f"{name:<20} {result['avg']:<10.1f} {result['max']:<10} {result['avg_time']:<10.1f}s")

    best_config = max(results.items(), key=lambda x: x[1]['avg'])
    print(f"Best MCTS config: {best_config[0]} with {best_config[1]['avg']:.1f} average")

    return results


def play_single_game(game, mcts, verbose=False, max_moves=1000):
    moves = 0

    while not game.is_game_over() and moves < max_moves:
        valid_actions = game.get_all_valid_actions()
        if not valid_actions:
            break

        if verbose:
            print(f"Move {moves + 1}, Score: {game.score}")

        action = mcts.search(game)
        if action is None:
            break

        shop_index = action // 64
        row = (action % 64) // 8
        col = action % 8

        success = game.place_shape(shop_index, row, col)
        if not success:
            if verbose:
                print("BŁĄD: MCTS wybrał nieprawidłowy ruch!")
            break

        moves += 1

    if verbose:
        print(f"Game finished! Score: {game.score}, Moves: {moves}")

    return game.score


def quick_test():
    print("Quick MCTS test:")

    game = GameSimulator()
    game.start(0)

    mcts = MCTS(c_param=1.4, time_limit=1.0)
    score = play_single_game(game, mcts, verbose=True)

    print(f"Final score: {score}")
    return score


if __name__ == "__main__":
    quick_test()