import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RLBaseline(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        return self.net(x)

    def act(self, observation):
        # observation: (D)
        probs = self(observation.unsqueeze(0)) # (1, 1)
        prob_commit = probs.squeeze()

        # Stochastic Policy
        m = torch.distributions.Bernoulli(prob_commit)
        action = m.sample()

        self.saved_log_probs.append(m.log_prob(action))

        return action.item() > 0.5, prob_commit.item()

class RLTrainer:
    def __init__(self, agent, lr=1e-3, gamma=0.99, device='cpu'):
        self.agent = agent.to(device)
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device

    def update(self):
        R = 0
        policy_loss = []
        returns = []

        # Calculate discounted returns from the end
        # Since we only get a terminal reward in this env,
        # The return for every step is just gamma^(T-t) * FinalReward
        # But wait, we get reward=0 usually.
        # Actually, self.agent.rewards contains the rewards for each step.
        # In this env, they are 0, 0, ..., Final.

        for r in self.agent.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for log_prob, R in zip(self.agent.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        if policy_loss:
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            self.optimizer.step()

        # Clear memory
        del self.agent.saved_log_probs[:]
        del self.agent.rewards[:]
        return loss.item() if policy_loss else 0.0
