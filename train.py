from GameSimulator import GameSimulator
from GameAI import GameAI
import numpy as np
from time import time

def train_ai():
    ai = GameAI()

    episodes = 10000
    scores = []
    t = time()

    for episode in range(episodes):
        game = GameSimulator()
        game.start()

        while not game.is_game_over():
            valid_actions = game.get_all_valid_actions()

            if not valid_actions:
                break

            state = game.get_state()
            action = ai.act(state, valid_actions)

            shop_index, row, col = action // 64, (action % 64) // 8, action % 8

            old_score = game.score
            success = game.place_shape(shop_index, row, col)

            if not success:
                print("nie udało się postawić kształtu")
                continue

            reward = game.score - old_score + game.combo + 0.1

            done = game.is_game_over()
            if done:
                reward -= 20 #kara za śmierć

            next_state = game.get_state()
            ai.remember(state, action, reward, next_state, done)

            if len(ai.memory) > ai.batch_size:
                ai.replay()

        scores.append(game.score)

        if episode % 50 == 0:
            if episode % 500 == 0:
                ai.update_target_network()
            recent_scores = scores[-50:]
            print(f"Episode {episode}, Avg: {np.mean(recent_scores):.1f}, Max: {max(recent_scores):.1f}, Min: {min(recent_scores):.1f}, Std: {np.std(recent_scores):.1f}, Epsilon: {ai.epsilon:.3f}, Time: {time() - t}")
            t = time()

    return ai, scores

if __name__ == "__main__":
    trained_ai, training_scores = train_ai()

    import torch
    torch.save(trained_ai.q_network.state_dict(), 'trained_model.pt')
    print("Model zapisany!")