from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from random import choice
from plansza import predicted_reward, penalty_for_losing
from GameSimulator import GameSimulator
from time import time
from numpy import mean, std

def best_action(state, valid_actions):
    best = []
    best_reward = float('-inf')
    for action in valid_actions:
        new_state = state.copy()
        success, lines_cleared = new_state.place_shape(action // 64, (action % 64) // 8, action % 8)
        if success:
            reward = predicted_reward(new_state.board, lines_cleared, new_state.shop, 0)
            if new_state.is_game_over():
                reward += penalty_for_losing
            if reward > best_reward:
                best = [action]
                best_reward = reward
            elif reward == best_reward:
                best.append(action)
        else:
            print("nic nie wybrano")
    return choice(best)

def run():
    t = time()
    game = GameSimulator()
    game.start(0)
    moves = 0
    while not game.is_game_over():
        valid_actions = game.get_all_valid_actions()

        if not valid_actions:
            break

        action = best_action(game, valid_actions)

        success, lines_cleared = game.place_shape(action // 64, (action % 64) // 8, action % 8)

        if not success:
            print("nie udało się postawić kształtu")
            continue
        moves += 1
    print(f"skończyłem po {time()-t:.1f}s")
    return game.score, moves

def loop():
    num_workers = max(1, cpu_count() - 10)
    loops = 10
    print(f"Używam {num_workers} procesów równoległych do {loops} gier")

    t = time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run) for i in range(loops)]
        scores = []
        moves = []
        for i, future in enumerate(futures):
            score, move = future.result()
            scores.append(score)
            moves.append(move)
    print(f"Avg: {mean(scores):.1f}, Avg Moves: {mean(moves):.1f}, Max: {max(scores)}, Min: {min(scores)}, Std: {std(scores):.1f}, Time: {time() - t:.1f}")

if __name__ == "__main__":
    loop()