from torch import device, nn, optim, cuda, save, load, FloatTensor, stack, no_grad
from torch.nn import functional as F
import random
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_size=238, hidden_size=1024, output_size=192):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.network(x)

class GameAI:
    def __init__(self, learning_rate=0.0001, memory_file="ai_training_state.pt"):
        self.device = device("cuda")
        self.policy_network  = PolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.episodes_data = []
        self.current_episode = []

        self.epsilon = 0.3
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.9995

        self.memory_file = memory_file
        self.baseline_scores = []
        self.load_training_state()

    def save_training_state(self):
        state = {
            'epsilon': self.epsilon,
            'baseline_scores': self.baseline_scores,
            'policy_network_state': self.policy_network.state_dict(),
            'episodes_data': self.episodes_data
        }
        save(state, self.memory_file)

    def load_training_state(self):
        try:
            state = load(self.memory_file, weights_only=False)
            self.epsilon = state.get('epsilon', self.epsilon)
            self.baseline_scores = state.get('baseline_scores', [])
            self.episodes_data = state.get('episodes_data', [])
            self.policy_network.load_state_dict(state['policy_network_state'])
        except FileNotFoundError:
            print("Nowy trening")

    def act(self, state, valid_actions):
        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
            state_tensor = FloatTensor(state).unsqueeze(0).to(self.device)
            masked_logits = self.policy_network(state_tensor).clone()
            for i in range(192):
                if i not in set(valid_actions):
                    masked_logits[0][i] = float('-inf')
            log_prob = F.log_softmax(masked_logits, dim=1)[0][action]
            return action, log_prob.detach()

        state_tensor = FloatTensor(state).unsqueeze(0).to(self.device)

        masked_logits = self.policy_network(state_tensor).clone()
        for i in range(192):  # 8x8 pozycji x 3 elementy w sklepie
            if i not in set(valid_actions):
                masked_logits[0][i] = float('-inf')

        action = F.softmax(masked_logits, dim=1).multinomial(1).item()
        log_prob = F.log_softmax(masked_logits, dim=1)[0][action]

        return action, log_prob

    def store_transition(self, state, action, log_prob):
        self.current_episode.append((state.copy(), action, log_prob))

    def finish_episode(self, final_score):
        if not self.current_episode:
            return
        self.baseline_scores.append(final_score)
        if len(self.baseline_scores) > 1000:
            self.baseline_scores.pop(0)

        baseline = np.mean(self.baseline_scores) if self.baseline_scores else 0
        advantage = final_score - baseline

        episode_data = []
        for state, action, log_prob in self.current_episode:
            episode_data.append((state, action, log_prob, advantage))
        self.episodes_data.append(episode_data)
        self.current_episode = []

    def update_policy(self, batch_size=32):
        if len(self.episodes_data) < batch_size:
            return

        recent_episodes = self.episodes_data[-batch_size:]
        policy_losses = []
        all_advantages = []

        for episode in recent_episodes:
            for _, _, _, advantage in episode:
                all_advantages.append(advantage)
        if not all_advantages:
            return

        advantages_mean = np.mean(all_advantages)
        advantages_std = np.std(all_advantages) + 1e-8

        for episode in recent_episodes:
            for state, action, log_prob, advantage in episode:
                normalized_advantage = (advantage - advantages_mean) / advantages_std
                policy_losses.append(-log_prob * normalized_advantage)
        if not policy_losses:
            return

        self.optimizer.zero_grad()
        total_loss = stack(policy_losses).mean()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.optimizer.step()

        if len(self.episodes_data) > batch_size * 2:
            self.episodes_data = self.episodes_data[-batch_size:]

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)