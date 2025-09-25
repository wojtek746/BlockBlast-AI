from GameSimulator import GameSimulator
from numpy import mean, std
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from torch import device as torch_device
from torch import FloatTensor, no_grad, save
from torch.nn import functional as F
import random
import numpy as np
from plansza import predicted_reward, penalty_for_losing

def run_single_episode(policy_network_state, epsilon, episode_num):
    from GameAI import PolicyNetwork

    device = torch_device("cpu")
    policy_network = PolicyNetwork().to(device)
    policy_network.load_state_dict({k: v.cpu() for k, v in policy_network_state.items()})
    policy_network.eval()

    def simple_act(state, valid_actions):
        if random.random() < epsilon:
            return random.choice(valid_actions)

        state_tensor = FloatTensor(state).unsqueeze(0)

        with no_grad():
            logits = policy_network(state_tensor)

        masked_logits = logits.clone()
        valid_set = set(valid_actions)
        for i in range(192):
            if i not in valid_set:
                masked_logits[0][i] = float('-inf')

        return F.softmax(masked_logits, dim=1).multinomial(1).item()

    game = GameSimulator()
    game.start(episode_num)
    predicted_reward(game.board, 0)

    episode_transitions = []  # (state, action, reward)

    while not game.is_game_over():
        valid_actions = game.get_all_valid_actions()

        if not valid_actions:
            break

        state = game.get_state()
        action = simple_act(state, valid_actions)

        shop_index, row, col = action // 64, (action % 64) // 8, action % 8

        success, lines_cleared = game.place_shape(shop_index, row, col)

        if not success:
            print("nie udało się postawić kształtu")
            continue

        reward = predicted_reward(game.board, lines_cleared)
        done = game.is_game_over()
        if done:
            reward += penalty_for_losing

        episode_transitions.append((state, action, reward))

    return episode_transitions, game.score

class PipelinedTrainer:
    def __init__(self, ai, num_workers):
        self.ai = ai
        self.num_workers = num_workers
        self.batch_episodes = num_workers * 5

        self.executor = ProcessPoolExecutor(max_workers=num_workers)

    def start_simulation_batch(self, episode_num):
        actor_network_state = {k: v.cpu() for k, v in self.ai.actor_network.state_dict().items()}
        current_epsilon = self.ai.epsilon

        futures = [self.executor.submit(run_single_episode, actor_network_state, current_epsilon, episode_num) for _ in range(self.batch_episodes)]

        return futures

    def collect_simulation_results(self, futures):
        all_transitions = []
        batch_scores = []

        for future in futures:
            episode_transitions, score = future.result()
            all_transitions.extend(episode_transitions)
            batch_scores.append(score)

        return all_transitions, batch_scores

    def train_on_batch(self, all_transitions):
        episodes = {}
        for state, action, reward, next_state, done, valid_actions in all_transitions:
            self.ai.store_transition(state, action, reward, next_state, done, valid_actions)
            if done:
                final_score = sum(t[2] for t in self.ai.episode_data)
                self.ai.finish_episode(final_score)

def train_ai():
    from GameAI import GameAI
    from time import time

    ai = GameAI()
    episodes = 1000000
    scores = []

    num_workers = max(1, cpu_count() - 10)
    trainer = PipelinedTrainer(ai, num_workers)
    print(f"Używam {num_workers} procesów równoległych")

    current_futures = trainer.start_simulation_batch(0)

    t = time()
    for episode_batch in range(0, episodes, trainer.batch_episodes):

        all_transitions, batch_scores = trainer.collect_simulation_results(current_futures)
        scores.extend(batch_scores)

        if episode_batch + trainer.batch_episodes < episodes:
            current_futures = trainer.start_simulation_batch(episode_batch + trainer.batch_episodes)
        trainer.train_on_batch(all_transitions)

        current_episode = episode_batch + trainer.batch_episodes
        ai.update_epsilon()

        if current_episode % 5000 == 0:
            ai.save_training_state()
            print("zapisano training do pliku")

        if current_episode % 1000 == 0:
            recent_scores = scores[-1000:] if len(scores) >= 1000 else scores
            better = [i for i in recent_scores if i > 1000]

            print(f"Episode: {current_episode}, Avg: {mean(recent_scores):.1f}, Avg dla > 100: {mean(better) if better else 0:.1f}, Max: {max(recent_scores):.1f}, Min: {min(recent_scores):.1f}, Std: {std(recent_scores):.1f}, Epsilon: {ai.epsilon:.5f}, Baseline: {ai.get_baseline():.1f}, Time: {(time() - t):.1f}")
            t = time()

        if current_episode % 10000 == 0:
            save({
                'actor': ai.actor_network.state_dict(),
                'critic': ai.critic_network.state_dict()
            }, 'trained_model.pt')
            print("zapisano model")

    trainer.executor.shutdown()
    return ai, scores

if __name__ == "__main__":
    trained_ai, training_scores = train_ai()