from GameSimulator import GameSimulator
from numpy import mean, std
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from torch import device as torch_device
from torch import FloatTensor, no_grad, save
from torch.nn import functional as F
import random
import numpy as np
from plansza import funkcja

def run_single_episode(actor_network_state, epsilon, episode_num):
    from GameAI import ActorNetwork

    device = torch_device("cpu")
    actor_network = ActorNetwork().to(device)
    actor_network.load_state_dict({k: v.cpu() for k, v in actor_network_state.items()})
    actor_network.eval()

    def simple_act(state, valid_actions):
        if random.random() < epsilon:
            return random.choice(valid_actions)

        state_tensor = FloatTensor(state).unsqueeze(0)

        with no_grad():
            logits = actor_network(state_tensor)

        masked_logits = logits.clone()
        valid_set = set(valid_actions)
        for i in range(192):
            if i not in valid_set:
                masked_logits[0][i] = float('-inf')

        action = F.softmax(masked_logits, dim=1).multinomial(1).item()

        return action

    game = GameSimulator()
    game.start(episode_num)

    episode_transitions = []  # (state, action, reward, next_state, done, valid_actions)

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

        reward = funkcja(game)
        done = game.is_game_over()
        next_state = game.get_state()

        episode_transitions.append((state, action, reward, next_state, done, valid_actions))

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
        for transition in all_transitions:
            self.ai.store_transition(*transition)
        self.ai.update_networks(batch_size=min(256, len(all_transitions)))

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

            empty_board_game = GameSimulator()
            empty_board_game.reload_shop()

            normal_start_game = GameSimulator()
            normal_start_game.start(current_episode)

            print(f"Episode: {current_episode}, Avg: {mean(recent_scores):.1f}, Avg dla > 100: {mean(better) if better else 0:.1f}, Max: {max(recent_scores):.1f}, Min: {min(recent_scores):.1f}, Std: {std(recent_scores):.1f}, Epsilon: {ai.epsilon:.5f}, Empty_Board+Shop: {ai.get_state_value(empty_board_game.get_state()):.1f}, Completely_Empty: {ai.get_state_value(GameSimulator().get_state()):.1f}, Normalny_Start: {ai.get_state_value(normal_start_game.get_state()):.1f}, Time: {(time() - t):.1f}")
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