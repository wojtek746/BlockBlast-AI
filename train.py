from GameSimulator import GameSimulator
from numpy import mean, std
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from torch import device as torch_device
from torch import FloatTensor, no_grad, save
from torch.nn import functional as F
import random

def run_single_episode(policy_network_state, epsilon, episode_num):
    from GameAI import PolicyNetwork

    device = torch_device("cpu")
    policy_network = PolicyNetwork().to(device)
    policy_network.load_state_dict({k: v.cpu() for k, v in policy_network_state.items()})
    policy_network.eval()

    def simple_act(state, valid_actions):
        if random.random() < epsilon:
            action = random.choice(valid_actions)
            state_tensor = FloatTensor(state).unsqueeze(0)
            with no_grad():
                logits = policy_network(state_tensor)

            masked_logits = logits.clone()
            for i in range(192):
                if i not in set(valid_actions):
                    masked_logits[0][i] = float('-inf')

            log_probs = F.log_softmax(masked_logits, dim=1)
            log_prob = log_probs[0][action]
            return action, log_prob

        state_tensor = FloatTensor(state).unsqueeze(0)

        with no_grad():
            logits = policy_network(state_tensor)

        masked_logits = logits.clone()
        for i in range(192):
            if i not in set(valid_actions):
                masked_logits[0][i] = float('-inf')

        action = F.softmax(masked_logits, dim=1).multinomial(1).item()
        log_prob = F.log_softmax(masked_logits, dim=1)[0][action]

        return action, log_prob

    game = GameSimulator()
    game.start(episode_num)

    episode_data = []

    while not game.is_game_over():
        valid_actions = game.get_all_valid_actions()

        if not valid_actions:
            break

        state = game.get_state()
        action, log_prob = simple_act(state, valid_actions)

        shop_index, row, col = action // 64, (action % 64) // 8, action % 8

        success = game.place_shape(shop_index, row, col)

        if not success:
            print("nie udało się postawić kształtu")
            continue

        episode_data.append((state.copy(), action, valid_actions.copy()))

    return episode_data, game.score

class PipelinedTrainer:
    def __init__(self, ai, num_workers):
        self.ai = ai
        self.num_workers = num_workers
        self.batch_episodes = num_workers * 4

        self.executor = ProcessPoolExecutor(max_workers=num_workers)

    def start_simulation_batch(self, episode_num):
        policy_network_state = {k: v.cpu() for k, v in self.ai.policy_network.state_dict().items()}
        current_epsilon = self.ai.epsilon

        futures = [self.executor.submit(run_single_episode, policy_network_state, current_epsilon, episode_num) for _ in range(self.batch_episodes)]

        return futures

    def collect_simulation_results(self, futures):
        batch_episodes_data = []
        batch_scores = []

        for future in futures:
            episode_data, score = future.result()
            batch_episodes_data.append((episode_data, score))
            batch_scores.append(score)

        return batch_episodes_data, batch_scores

    def train_on_batch(self, batch_episodes_data):
        for episode_data, final_score in batch_episodes_data:
            for state, action, valid_actions in episode_data:
                self.ai.store_transition(state, action, valid_actions)
            self.ai.finish_episode(final_score)
        self.ai.update_policy(batch_size=len(batch_episodes_data))

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

        batch_episodes_data, batch_scores = trainer.collect_simulation_results(current_futures)
        scores.extend(batch_scores)

        if episode_batch + trainer.batch_episodes < episodes:
            current_futures = trainer.start_simulation_batch(episode_batch + trainer.batch_episodes)
        trainer.train_on_batch(batch_episodes_data)

        current_episode = episode_batch + trainer.batch_episodes
        ai.update_epsilon()

        if current_episode % 5000 == 0:
            ai.save_training_state()
            print("zapisano training do pliku")

        if current_episode % 100 == 0:
            recent_scores = scores[-100:] if len(scores) >= 100 else scores
            better = [i for i in recent_scores if i > 100]
            baseline = mean(ai.baseline_scores) if ai.baseline_scores else 0
            print(f"Episode: {current_episode}, Avg: {mean(recent_scores):.1f}, Avg dla > 100: {mean(better) if better else 0:.1f}, Max: {max(recent_scores):.1f}, Min: {min(recent_scores):.1f}, Std: {std(recent_scores):.1f}, Epsilon: {ai.epsilon:.5f}, Baseline: {baseline:.1f}, Time: {(time() - t):.1f}")
            t = time()

        if current_episode % 10000 == 0:
            save(ai.policy_network.state_dict(), 'trained_model.pt')
            print("zapisano model")

    trainer.executor.shutdown()
    return ai, scores

if __name__ == "__main__":
    trained_ai, training_scores = train_ai()

    save(trained_ai.policy_network.state_dict(), 'trained_model.pt')
    trained_ai.save_training_state()
    print("Model zapisany!")