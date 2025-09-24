from torch import device, nn, optim, cuda, save, load, FloatTensor, stack, LongTensor, BoolTensor
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size=238, hidden_size=1024, output_size=192):
        super(DQN, self).__init__()
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
    def __init__(self, learning_rate=0.000005, memory_file="ai_training_state.pt"):
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.memory = deque(maxlen=100000)
        self.epsilon = 0
        self.epsilon_min = 0
        self.batch_size = 512
        self.gamma = 0.5
        self.memory_file = memory_file
        self.load_training_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save_training_state(self):
        state = {
            'epsilon': self.epsilon,
            'memory': list(self.memory),
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict()
        }
        save(state, self.memory_file)

    def load_training_state(self):
        try:
            state = load(self.memory_file, weights_only=False)
            self.epsilon = state['epsilon']
            self.memory.extend(state['memory'])
            self.q_network.load_state_dict(state['q_network_state'])
            self.target_network.load_state_dict(state['target_network_state'])
        except FileNotFoundError:
            print("Nowy trening")

    def act(self, state, valid_actions):
        if random.random() <= self.epsilon:
            return random.choice(valid_actions)

        state_tensor = FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)

        masked_q_values = q_values.clone()
        valid_indices = set(valid_actions)
        for i in range(192):  # 8x8 pozycji x 3 elementy w sklepie
            if i not in valid_indices:
                masked_q_values[0][i] = float('-inf')

        return masked_q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states = stack([FloatTensor(e[0]) for e in batch]).to(self.device)
        actions = LongTensor([e[1] for e in batch]).to(self.device)
        rewards = FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = stack([FloatTensor(e[3]) for e in batch]).to(self.device)
        dones = BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5) #nie wiem, co to robi xd
        self.optimizer.step()

    def update_target_network(self, tau=0.001):
        self.target_network.load_state_dict(self.q_network.state_dict())
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)