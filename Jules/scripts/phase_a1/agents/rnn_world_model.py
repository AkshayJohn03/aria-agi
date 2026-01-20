import torch
import torch.nn as nn
import torch.optim as optim

class RNNWorldModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

        self.reset_memory()

    def reset_memory(self):
        self.hidden = None

    def forward(self, x, hidden=None):
        # x: (B, T, D)
        out, new_hidden = self.rnn(x, hidden)
        pred = torch.sigmoid(self.head(out))
        return pred, new_hidden

    def act(self, observation, threshold=0.5):
        # observation: (D)
        input_tensor = observation.view(1, 1, -1) # (1, 1, D)

        with torch.no_grad():
            pred, self.hidden = self(input_tensor, self.hidden)
            hazard_prob = pred.item()

        return hazard_prob > threshold, hazard_prob

class RNNTrainer:
    def __init__(self, agent, lr=1e-3, device='cpu'):
        self.agent = agent.to(device)
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        self.device = device
        self.criterion = nn.BCELoss()

    def train_on_batch(self, batch_episodes):
        self.agent.train()
        total_loss = 0

        for ep in batch_episodes:
            obs_seq = torch.stack([x[0] for x in ep]).to(self.device).unsqueeze(0) # (1, T, D)
            hazards = torch.tensor([1.0 if x[1] else 0.0 for x in ep]).to(self.device)

            # Targets: Hazard in next 20 steps
            targets = []
            for t in range(len(hazards)):
                future = hazards[t+1 : t+21]
                targets.append(1.0 if future.sum() > 0 else 0.0)
            targets = torch.tensor(targets).to(self.device).view(1, -1, 1) # (1, T, 1)

            preds, _ = self.agent(obs_seq) # (1, T, 1)

            loss = self.criterion(preds, targets)
            loss.backward()
            total_loss += loss.item()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss / len(batch_episodes)
