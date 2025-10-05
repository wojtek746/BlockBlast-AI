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

        # valid_vals = logits[0, valid_actions]
        # invalid_idxs = [i for i in range(192) if i not in valid_set]
        # invalid_vals = logits[0, invalid_idxs]

        # print("Valid mean:", valid_vals.mean().item(), "std:", valid_vals.std().item())
        # print("Invalid mean:", invalid_vals.mean().item(), "std:", invalid_vals.std().item())
        # print()

        return F.softmax(masked_logits, dim=1).multinomial(1).item()

    game = GameSimulator()
    game.start(episode_num)
    predicted_reward(game.board, 0, game.shop)

    episode_transitions = []  # (state, action, reward, valid_actions)

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

        reward = predicted_reward(game.board, lines_cleared, game.shop)
        done = game.is_game_over()
        if done:
            reward += penalty_for_losing

        episode_transitions.append((state, action, reward, valid_actions))

    return episode_transitions, game.score

class PipelinedTrainer:
    def __init__(self, ai, num_workers):
        self.ai = ai
        self.num_workers = num_workers
        self.batch_episodes = num_workers * 5

        self.executor = ProcessPoolExecutor(max_workers=num_workers)

    def start_simulation_batch(self, episode_num):
        policy_network_state = {k: v.cpu() for k, v in self.ai.policy_network.state_dict().items()}
        current_epsilon = self.ai.epsilon

        futures = [self.executor.submit(run_single_episode, policy_network_state, current_epsilon, episode_num) for _ in range(self.batch_episodes)]

        return futures

    def collect_simulation_results(self, futures):
        all_episodes = []
        batch_scores = []

        for future in futures:
            episode_transitions, score = future.result()
            all_episodes.append((episode_transitions, score))
            batch_scores.append(score)

        return all_episodes, batch_scores

    def train_on_batch(self, all_episodes):
        all_scores = [final_score for _, final_score in all_episodes]
        self.ai.baseline_scores.extend(all_scores)
        if len(self.ai.baseline_scores) > 1000:
            self.ai.baseline_scores = self.ai.baseline_scores[-1000:]

        baseline = np.mean(self.ai.baseline_scores) if self.ai.baseline_scores else 0

        all_states = []
        all_actions = []
        all_advantages = []
        all_valid_actions = []

        for episode_transitions, final_score in all_episodes:
            advantage = final_score - baseline

            for state, action, reward, valid_actions in episode_transitions:
                all_states.append(state)
                all_actions.append(action)
                all_advantages.append(advantage)
                all_valid_actions.append(valid_actions)

        if not all_states:
            return

        max_batch_size = 50
        if len(all_states) > max_batch_size:
            indices = random.sample(range(len(all_states)), max_batch_size)
            all_states = [all_states[i] for i in indices]
            all_actions = [all_actions[i] for i in indices]
            all_advantages = [all_advantages[i] for i in indices]
            all_valid_actions = [all_valid_actions[i] for i in indices]

        self.ai.batch_update(all_states, all_actions, all_advantages, all_valid_actions)

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

        all_episodes, batch_scores = trainer.collect_simulation_results(current_futures)
        scores.extend(batch_scores)

        if episode_batch + trainer.batch_episodes < episodes:
            current_futures = trainer.start_simulation_batch(episode_batch + trainer.batch_episodes)
        trainer.train_on_batch(all_episodes)

        current_episode = episode_batch + trainer.batch_episodes
        ai.update_epsilon()

        if current_episode % 5000 == 0:
            ai.save_training_state()
            print("zapisano training do pliku")

        how_often = 1000
        if current_episode % how_often == 0:
            recent_scores = scores[-how_often:] if len(scores) >= how_often else scores
            better = [i for i in recent_scores if i > 100]

            print(f"Episode: {current_episode}, Avg: {mean(recent_scores):.1f}, Avg dla > 100: {mean(better) if better else 0:.1f}, Max: {max(recent_scores):.1f}, Min: {min(recent_scores):.1f}, Std: {std(recent_scores):.1f}, Epsilon: {ai.epsilon:.5f}, Baseline: {(np.mean(ai.baseline_scores) if ai.baseline_scores else 0):.1f}, Time: {(time() - t):.1f}")
            t = time()

        if current_episode % 10000 == 0:
            save(ai.policy_network.state_dict(), 'trained_model.pt')
            print("zapisano model")

    trainer.executor.shutdown()
    return ai, scores

if __name__ == "__main__":
    trained_ai, training_scores = train_ai()