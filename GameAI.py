from torch import device, nn, optim, cuda, save, load, FloatTensor, stack, no_grad, LongTensor
from torch.nn import functional as F
import random
import numpy as np

class PolicyNetwork(nn.Module): #wielkość sieci, że ile neuronów ma mieć
    def __init__(self, input_size=171, hidden_size=2048, output_size=192):
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

        self.policy_network = PolicyNetwork().to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.episode_data = []  # (state, action, log_prob, reward)
        self.baseline_scores = []

        self.epsilon = 1
        self.epsilon_min = 0.00001
        self.epsilon_max = 1
        self.epsilon_decay = 0.999

        self.memory_file = memory_file
        self.load_training_state()

    def save_training_state(self):
        state = {
            'epsilon': self.epsilon,
            'policy_network_state': self.policy_network.state_dict(),
            'baseline_scores': self.baseline_scores[-1000:] if len(self.baseline_scores) > 1000 else self.baseline_scores
        }
        save(state, self.memory_file)

    def load_training_state(self):
        try:
            state = load(self.memory_file, weights_only=False)
            self.epsilon = state.get('epsilon', self.epsilon)
            self.baseline_scores = state.get('baseline_scores', [])
            self.policy_network.load_state_dict(state['policy_network_state'])
        except FileNotFoundError:
            print("Nowy trening")

    def act(self, state, valid_actions):
        if random.random() < self.epsilon:
            action = random.choice(valid_actions)

            state_tensor = FloatTensor(state).unsqueeze(0).to(self.device)

            with no_grad():
                logits = self.policy_network(state_tensor)

            masked_logits = logits.clone()
            valid_set = set(valid_actions)
            for i in range(192):  # 8x8 pozycji x 3 elementy w sklepie
                if i not in valid_set:
                    masked_logits[0][i] = float('-inf')

            log_prob = F.log_softmax(masked_logits, dim=1)[0][action]
            return action, log_prob.detach()
        state_tensor = FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy_network(state_tensor)

        masked_logits = logits.clone()
        valid_set = set(valid_actions)
        for i in range(192):
            if i not in valid_set:
                masked_logits[0][i] = float('-inf')

        action = F.softmax(masked_logits, dim=1).multinomial(1).item()
        log_prob = F.log_softmax(masked_logits, dim=1)[0][action]

        return action, log_prob

    def finish_episode(self, final_score):
        if not self.episode_data:
            return

        self.baseline_scores.append(final_score)
        if len(self.baseline_scores) > 10:
            self.baseline_scores.pop(0)

        baseline = np.mean(self.baseline_scores) if self.baseline_scores else 0
        advantage = final_score - baseline
        policy_losses = []

        for state, action, reward, valid_actions in self.episode_data:
            state_tensor = FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.policy_network(state_tensor)

            masked_logits = logits.clone()
            valid_set = set(valid_actions)
            for i in range(192):
                if i not in valid_set:
                    masked_logits[0][i] = float('-inf')

            log_prob = F.log_softmax(masked_logits, dim=1)[0][action]
            policy_losses.append(-log_prob * advantage)

        if policy_losses:
            self.optimizer.zero_grad()
            total_loss = stack(policy_losses).mean()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.optimizer.step()
        self.episode_data = []

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_max

    def batch_update(self, states, actions, advantages, valid_actions_list):
        if not states:
            return
        advantages = np.array(advantages)
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        states_tensor = FloatTensor(np.array(states)).to(self.device)
        actions_tensor = LongTensor(actions).to(self.device)
        advantages_tensor = FloatTensor(advantages).to(self.device)

        logits = self.policy_network(states_tensor)

        masked_logits = logits.clone()
        for batch_idx, valid_actions in enumerate(valid_actions_list):
            valid_set = set(valid_actions)
            for action_idx in range(192):
                if action_idx not in valid_set:
                    masked_logits[batch_idx][action_idx] = float('-inf')

        selected_log_probs = F.log_softmax(logits, dim=1).gather(1, actions_tensor.unsqueeze(1)).squeeze()
        policy_loss = -(selected_log_probs * advantages_tensor).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.optimizer.step()