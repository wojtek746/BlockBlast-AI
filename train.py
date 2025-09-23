from GameSimulator import GameSimulator
from numpy import mean, std
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from torch import device as torch_device
from torch import FloatTensor, no_grad, save
from queue import Queue

def run_single_episode(q_network_state, epsilon, episode_num):
    from GameAI import DQN
    import random
    from collections import deque

    device = torch_device("cpu")
    q_network = DQN().to(device)
    q_network.load_state_dict({k: v.cpu() for k, v in q_network_state.items()})
    q_network.eval()

    def simple_act(state, valid_actions):
        if random.random() <= epsilon:
            return random.choice(valid_actions)
        state_tensor = FloatTensor(state).unsqueeze(0)

        with no_grad():
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
    recent_moves_buffer = deque(maxlen=5)

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

        reward = game.score - old_score

        done = game.is_game_over()

        next_state = game.get_state()

        move_data = {
            'state': state.copy(),
            'action': action,
            'original_reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'episode_memory_index': len(episode_memory)  # Index w episode_memory
        }
        recent_moves_buffer.append(move_data)

        episode_memory.append((state, action, reward, next_state, done))

        if done: #kara za śmierć
            death_penalties = [10, 9, 8, 7, 6]
            for i, move_data in enumerate(recent_moves_buffer):
                penalty_index = len(recent_moves_buffer) - 1 - i
                death_penalty = death_penalties[penalty_index]
                original_reward = move_data['original_reward']
                new_reward = original_reward - death_penalty
                memory_index = move_data['episode_memory_index']
                old_entry = episode_memory[memory_index]

                episode_memory[memory_index] = (
                    old_entry[0],
                    old_entry[1],
                    new_reward,
                    old_entry[3],
                    old_entry[4]
                )
            break

    return episode_memory, game.score


class PipelinedTrainer:
    def __init__(self, ai, num_workers):
        self.ai = ai
        self.num_workers = num_workers
        self.batch_episodes = num_workers * 10

        self.simulation_queue = Queue(maxsize=2)
        self.training_queue = Queue(maxsize=2)

        self.executor = ProcessPoolExecutor(max_workers=num_workers)

    def start_simulation_batch(self, episode_num):
        q_network_state = {k: v.cpu() for k, v in self.ai.q_network.state_dict().items()}
        current_epsilon = self.ai.epsilon

        futures = [self.executor.submit(run_single_episode, q_network_state, current_epsilon, episode_num) for _ in range(self.batch_episodes)]

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
            training_steps = 5

            for _ in range(training_steps):
                self.ai.replay()

def update_epsilon(ai, episode):
    epsilon_decay_episodes = 40000
    if episode < epsilon_decay_episodes:
        decay_rate = 0.9995
        ai.epsilon = max(ai.epsilon_min, ai.epsilon * decay_rate)
    else:
        ai.epsilon = ai.epsilon_min

def train_ai():
    from GameAI import GameAI
    from time import time

    ai = GameAI()
    episodes = 100000
    scores = []

    num_workers = max(1, cpu_count() - 10)
    trainer = PipelinedTrainer(ai, num_workers)
    print(f"Używam {num_workers} procesów równoległych")

    current_futures = trainer.start_simulation_batch(0)

    t = time()
    for episode_batch in range(0, episodes, trainer.batch_episodes):

        batch_memories, batch_scores = trainer.collect_simulation_results(current_futures)
        scores.extend(batch_scores)

        if episode_batch + trainer.batch_episodes < episodes:
            current_futures = trainer.start_simulation_batch(episode_batch + trainer.batch_episodes)
        trainer.train_on_batch(batch_memories)

        current_episode = episode_batch + trainer.batch_episodes
        update_epsilon(ai, current_episode)

        if current_episode % 5000 == 0:
            ai.update_target_network()
            ai.save_training_state()
            print("zapisano memory do pliku")

        if current_episode % 100 == 0:
            recent_scores = scores[-100:] if len(scores) >= 100 else scores
            better = []
            for i in recent_scores:
                if i > 100:
                    better.append(i)
            print(f"Episode: {current_episode}, Avg: {mean(recent_scores):.1f}, Avg dla > 100: {mean(better):.1f}, Max: {max(recent_scores):.1f}, Min: {min(recent_scores):.1f}, Std: {std(recent_scores):.1f}, Epsilon: {ai.epsilon:.5f}, Time: {(time() - t):.1f}")
            t = time()

        if current_episode % 10000 == 0:
            save(ai.q_network.state_dict(), 'trained_model.pt')
            print("zapisano model")

    trainer.executor.shutdown()
    return ai, scores

if __name__ == "__main__":
    trained_ai, training_scores = train_ai()

    save(trained_ai.q_network.state_dict(), 'trained_model.pt')
    trained_ai.save_training_state()
    print("Model zapisany!")