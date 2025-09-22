from sympy.physics.units import action

from GameSimulator import GameSimulator
from GameAI import GameAI
import numpy as np

def train_ai():
    ai = GameAI()

    episodes = 10000
    scores = []

    for episode in range(episodes):
        game = GameSimulator()

        while not game.game_over:
            valid_actions = game.get_all_valid_actions()

            if not valid_actions:
                game.game_over = True
                break

            state = game.get_state()
            action = ai.act(state, valid_actions)

            shop_index, row, col = action // 64, (action % 64) // 8, action % 8

            old_score = game.score
            game.place_shape(shop_index, row, col)

            reward = game.score - old_score

            if game.is_game_over():
                game.game_over = True
                reward -= 100

            next_state = game.get_state()
            ai.remember(state, action, reward, next_state, game.game_over)

        scores.append(game.score)
        ai.replay()

        if episode % 100 == 0:
            ai.update_target_network()
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {ai.epsilon:.3f}")

    return ai, scores

if __name__ == "__main__":
    trained_ai, training_scores = train_ai()