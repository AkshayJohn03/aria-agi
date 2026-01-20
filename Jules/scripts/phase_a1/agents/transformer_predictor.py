import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=4, d_model=64, n_head=2, n_layer=2, history_len=20, max_seq_len=100):
        super().__init__()
        self.history_len = history_len
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim

        # Embedding for continuous input
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # Head: Predicts Hazard Probability (Scalar [0, 1])
        self.head = nn.Linear(d_model, 1)

        # Buffer for the current episode
        self.reset_memory()

    def reset_memory(self):
        self.obs_history = []

    def forward(self, x):
        # x: (Batch, Seq, Input)
        b, t, _ = x.shape

        # Projection
        h = self.input_proj(x) # (B, T, D)

        # Positional Encoding (truncated to current seq len)
        if t <= self.max_seq_len:
            h = h + self.pos_embed[:, :t, :]
        else:
             # Fallback if seq is too long (shouldn't happen with correct max_len)
            h = h + self.pos_embed[:, :self.max_seq_len, :]

        # Attention
        out = self.transformer(h)

        # We only care about the prediction at the last step
        last_step = out[:, -1, :] # (B, D)

        pred = torch.sigmoid(self.head(last_step))
        return pred

    def act(self, observation, threshold=0.5):
        """
        observation: Tensor (Input Dim)
        """
        # Add to history
        self.obs_history.append(observation)

        # Truncate history
        if len(self.obs_history) > self.history_len:
            self.obs_history.pop(0)

        # Create tensor batch
        input_seq = torch.stack(self.obs_history).unsqueeze(0) # (1, T, D)

        with torch.no_grad():
            hazard_prob = self(input_seq).item()

        return hazard_prob > threshold, hazard_prob

class TransformerTrainer:
    def __init__(self, agent, lr=1e-3, device='cpu'):
        self.agent = agent.to(device)
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        self.device = device
        self.criterion = nn.BCELoss()

    def train_on_batch(self, batch_episodes):
        """
        batch_episodes: List of episodes.
        Each episode is a list of (obs, hazard_active_bool).
        """
        self.agent.train()
        total_loss = 0

        for ep in batch_episodes:
            obs_seq = torch.stack([x[0] for x in ep]).to(self.device) # (T, D)
            hazards = torch.tensor([1.0 if x[1] else 0.0 for x in ep]).to(self.device) # (T)

            # Generate Targets: Predict hazard in future window (e.g., next 20 steps)
            targets = []
            for t in range(len(hazards)):
                # Look ahead 20 steps
                future = hazards[t+1 : t+21]
                if future.sum() > 0:
                    targets.append(1.0)
                else:
                    targets.append(0.0)
            targets = torch.tensor(targets).to(self.device)

            seq_len = obs_seq.size(0)
            inp = obs_seq.unsqueeze(0) # (1, T, D)

            # Create causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(self.device)

            # Embed
            h = self.agent.input_proj(inp)
            h = h + self.agent.pos_embed[:, :seq_len, :]

            # Transformer with mask
            out = self.agent.transformer(h, mask=mask) # (1, T, D)

            preds = torch.sigmoid(self.agent.head(out)).squeeze() # (T)

            loss = self.criterion(preds, targets)
            loss.backward()
            total_loss += loss.item()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return total_loss / len(batch_episodes)
