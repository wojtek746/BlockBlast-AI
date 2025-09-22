from GameSimulator import GameSimulator
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import torch
from queue import Queue
import threading

def run_single_episode(q_network_state, epsilon, episode_num):
    from GameAI import DQN
    import random

    device = torch.device("cpu")
    q_network = DQN().to(device)
    q_network.load_state_dict({k: v.cpu() for k, v in q_network_state.items()})
    q_network.eval()

    def simple_act(state, valid_actions):
        if random.random() <= epsilon:
            return random.choice(valid_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = q_network(state_tensor)

        masked_q_values = q_values.clone()
        valid_set = set(valid_actions)
        for i in range(192):
            if i not in valid_set:
                masked_q_values[0][i] = float('-inf')

        return masked_q_values.argmax().item()

    game = GameSimulator()
    game.start(episode_num)

    episode_memory = []

    while not game.is_game_over():
        valid_actions = game.get_all_valid_actions()

        if not valid_actions:
            break

        state = game.get_state()
        action = simple_act(state, valid_actions)

        shop_index, row, col = action // 64, (action % 64) // 8, action % 8

        old_score = game.score
        success = game.place_shape(shop_index, row, col)

        if not success:
            print("nie udało się postawić kształtu")
            continue

        reward = game.score - old_score + min(game.combo * 0.1, 2.0) + 0.05

        done = game.is_game_over()
        if done:
            reward -= 10  # kara za śmierć

        next_state = game.get_state()
        episode_memory.append((state, action, reward, next_state, done))

    return episode_memory, game.score


class PipelinedTrainer:
    def __init__(self, ai, num_workers):
        self.ai = ai
        self.num_workers = num_workers
        self.batch_episodes = num_workers * 10

        self.simulation_queue = Queue(maxsize=2)
        self.training_queue = Queue(maxsize=2)

        self.executor = ProcessPoolExecutor(max_workers=num_workers)

    def start_simulation_batch(self, batch_id):
        q_network_state = {k: v.cpu() for k, v in self.ai.q_network.state_dict().items()}
        current_epsilon = self.ai.epsilon

        futures = [self.executor.submit(run_single_episode, q_network_state, current_epsilon) for _ in range(self.batch_episodes)]

        return futures

    def collect_simulation_results(self, futures):
        batch_memories = []
        batch_scores = []

        for future in futures:
            episode_memory, score = future.result()
            batch_memories.extend(episode_memory)
            batch_scores.append(score)

        return batch_memories, batch_scores

    def train_on_batch(self, batch_memories):
        for memory in batch_memories:
            self.ai.remember(*memory)

        if len(self.ai.memory) > self.ai.batch_size:
            training_steps = 20

            for _ in range(training_steps):
                self.ai.replay()

def update_epsilon(ai, episode):
    epsilon_decay_episodes = 50000
    if episode < epsilon_decay_episodes:
        progress = episode / epsilon_decay_episodes
        ai.epsilon = max(ai.epsilon_min, 1.0 - progress * (1.0 - ai.epsilon_min))
    else:
        ai.epsilon = ai.epsilon_min

def train_ai():
    from GameAI import GameAI
    from time import time

    ai = GameAI()
    episodes = 100000
    scores = []

    num_workers = max(1, mp.cpu_count() - 10)
    trainer = PipelinedTrainer(ai, num_workers)
    print(f"Używam {num_workers} procesów równoległych")

    current_futures = trainer.start_simulation_batch(0)

    for episode_batch in range(0, episodes, trainer.batch_episodes):
        t = time()

        batch_memories, batch_scores = trainer.collect_simulation_results(current_futures)
        scores.extend(batch_scores)

        if episode_batch + trainer.batch_episodes < episodes:
            current_futures = trainer.start_simulation_batch(episode_batch + trainer.batch_episodes)
        trainer.train_on_batch(batch_memories)

        current_episode = episode_batch + trainer.batch_episodes
        update_epsilon(ai, current_episode)

        if current_episode % 2000 == 0:
            ai.update_target_network()
            ai.save_training_state()

        if current_episode % 100 == 0:
            recent_scores = scores[-100:] if len(scores) >= 100 else scores
            print(f"Episode: {current_episode}, Avg: {np.mean(recent_scores):.1f}, Max: {max(recent_scores):.1f}, Min: {min(recent_scores):.1f}, Std: {np.std(recent_scores):.1f}, Epsilon: {ai.epsilon:.5f}, Time: {(time() - t):.1f}")
            t = time()

    trainer.executor.shutdown()
    return ai, scores

if __name__ == "__main__":
    trained_ai, training_scores = train_ai()

    torch.save(trained_ai.q_network.state_dict(), 'trained_model.pt')
    trained_ai.save_training_state()
    print("Model zapisany!")