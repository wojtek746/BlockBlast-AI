from plansza import predicted_reward, penalty_for_losing
from GameSimulator import GameSimulator
from time import time

def best_action(state, valid_actions):
    best = None
    best_reward = -1
    for action in valid_actions:
        new_state = state.copy()
        success, lines_cleared = new_state.place_shape(action // 64, (action % 64) // 8, action % 8)
        if success:
            reward = predicted_reward(new_state.board, lines_cleared, 0)
            if reward > best_reward:
                best = action
                best_reward = reward
    return best

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
    print(f"Score: {game.score}, Moves: {moves}, Time: {time() - t}")

if __name__ == "__main__":
    run()