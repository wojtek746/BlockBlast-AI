from torch import device, nn, optim, cuda, save, load, FloatTensor, stack, no_grad
from torch.nn import functional as F
import random
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, input_size=238, hidden_size=2048, output_size=192):
        super(ActorNetwork, self).__init__()
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

class CriticNetwork(nn.Module):
    def __init__(self, input_size=238, hidden_size=2048):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.network(x)

class GameAI:
    def __init__(self, learning_rate=0.0001, memory_file="ai_training_state.pt"):
        self.device = device("cuda")

        self.actor_network = ActorNetwork().to(self.device)
        self.critic_network = CriticNetwork().to(self.device)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=learning_rate * 10)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=learning_rate * 0.3)

        self.transitions = []  # (state, action, reward, next_state, done, valid_actions)

        self.epsilon = 0.5
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.999
        self.gamma = 0.8

        self.memory_file = memory_file
        self.load_training_state()

    def save_training_state(self):
        state = {
            'epsilon': self.epsilon,
            'actor_network_state': self.actor_network.state_dict(),
            'critic_network_state': self.critic_network.state_dict(),
            'transitions': self.transitions[-1000:] if len(self.transitions) > 1000 else self.transitions
        }
        save(state, self.memory_file)

    def load_training_state(self):
        try:
            state = load(self.memory_file, weights_only=False)
            self.epsilon = state.get('epsilon', self.epsilon)
            self.epsilon = 0.5
            self.transitions = state.get('transitions', [])
            self.actor_network.load_state_dict(state['actor_network_state'])
            self.critic_network.load_state_dict(state['critic_network_state'])
        except FileNotFoundError:
            print("Nowy trening")

    def act(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_tensor = FloatTensor(state).unsqueeze(0).to(self.device)

        with no_grad():
            logits = self.actor_network(state_tensor)

        masked_logits = logits.clone()
        for i in range(192):  # 8x8 pozycji x 3 elementy w sklepie
            if i not in set(valid_actions):
                masked_logits[0][i] = float('-inf')

        action = F.softmax(masked_logits, dim=1).multinomial(1).item()

        return action

    def store_transition(self, state, action, reward, next_state, done, valid_actions):
        self.transitions.append((state.copy(), action, reward, next_state.copy(), done, valid_actions.copy()))

    def update_networks(self, batch_size=128):
        if len(self.transitions) < batch_size:
            return

        recent_transitions = self.transitions[-batch_size:]

        actor_losses = []
        critic_losses = []
        advantages = []

        for state, action, reward, next_state, done, valid_actions in recent_transitions:
            state_tensor = FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = FloatTensor(next_state).unsqueeze(0).to(self.device)
            current_value = self.critic_network(state_tensor).squeeze()
            with no_grad():
                if done:
                    next_value = 0.0
                else:
                    next_value = self.critic_network(next_state_tensor).squeeze().item()
                td_target = reward + self.gamma * next_value
                advantage = td_target - current_value.item()
            advantages.append(advantage)

            critic_loss = F.mse_loss(current_value.unsqueeze(0), FloatTensor([td_target]).to(self.device))
            critic_losses.append(critic_loss)

            masked_logits = self.actor_network(state_tensor).clone()
            valid_set = set(valid_actions)
            for i in range(192):
                if i not in valid_set:
                    masked_logits[0][i] = float('-inf')

            log_probs = F.log_softmax(masked_logits, dim=1)
            log_prob = log_probs[0][action]
            actor_losses.append(-log_prob * advantage)

        if not actor_losses or not critic_losses:
            return

        if len(advantages) > 1:
            advantages = np.array(advantages)

            actor_losses = []
            idx = 0
            for state, action, reward, next_state, done, valid_actions in recent_transitions:
                state_tensor = FloatTensor(state).unsqueeze(0).to(self.device)

                masked_logits = self.actor_network(state_tensor).clone()
                valid_set = set(valid_actions)
                for i in range(192):
                    if i not in valid_set:
                        masked_logits[0][i] = float('-inf')

                log_probs = F.log_softmax(masked_logits, dim=1)
                log_prob = log_probs[0][action]

                normalized_advantage = (advantages[idx] - np.mean(advantages)) / (np.std(advantages) + 1e-8)

                actor_loss = -log_prob * normalized_advantage
                actor_losses.append(actor_loss)
                idx += 1

        self.actor_optimizer.zero_grad()
        total_actor_loss = stack(actor_losses).mean()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_network.parameters(), 0.5)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        total_critic_loss = stack(critic_losses).mean()
        total_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_network.parameters(), 0.5)
        self.critic_optimizer.step()

        if len(self.transitions) > batch_size * 3:
            self.transitions = self.transitions[-batch_size * 2:]

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_state_value(self, state):
        state_tensor = FloatTensor(state).unsqueeze(0).to(self.device)
        with no_grad():
            return self.critic_network(state_tensor).item()