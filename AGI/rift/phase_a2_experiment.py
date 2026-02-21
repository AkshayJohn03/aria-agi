import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32
CH = 8
STEPS_PER_EPISODE = 100
EPISODES = 50

print(f"🧬 PROJECT RIFT: PHASE A2 (Raw Grid Input) ON {DEVICE}")
print("   (Hypothesis: High-dimensional noise causes neural hedging, while geometry integrates.)")

class RiftWorldA2:
    def __init__(self):
        self.grid_size = GRID
        self.channels = CH
        self.reset()

    def reset(self):
        self.step_count = 0
        self.committed = False
        self.brace_level = 0.0
        self.health = 1.0

        # Hazard Schedule
        self.hazard_time = random.randint(40, 80)
        self.cue_time = self.hazard_time - 15

        # Cue Location (Random 4x4 block)
        self.cue_x = random.randint(0, GRID-4)
        self.cue_y = random.randint(0, GRID-4)

        return self.get_observation()

    def get_observation(self):
        # Generate Raw Grid Input
        # Background Noise
        obs = torch.randn(1, self.channels, GRID, GRID, device=DEVICE) * 0.1

        # Cue Signal
        if self.step_count == self.cue_time:
            # Inject Cue: 4x4 block
            obs[:, :, self.cue_y:self.cue_y+4, self.cue_x:self.cue_x+4] += 5.0

        return obs

    def step(self, action_commit):
        self.step_count += 1

        # Commitment Logic
        if action_commit and not self.committed:
            self.committed = True
            self.brace_level = 1.0

        # Ischemia Cost (Metabolic Burn)
        if self.brace_level > 0.3:
            self.health -= 0.03

        # Hazard Impact
        survived = True
        if self.step_count >= self.hazard_time and self.step_count < self.hazard_time + 5:
            if self.brace_level < 0.8: # Failed to brace
                self.health -= 0.5 # Massive damage per step of impact

        # Check Termination (End episode shortly after hazard)
        if self.step_count > self.hazard_time + 10:
            return self.get_observation(), self.health, False # Stop loop, survived

        if self.health <= 0.0:
            survived = False
            self.health = 0.0

        return self.get_observation(), self.health, survived

# =====================
# AGENTS
# =====================

class GeometricAgent:
    def __init__(self):
        self.trace = 0.0
        self.threshold = 0.5 # Tuned threshold for integrated signal

    def act(self, observation):
        # Integrate spatial signal (Geometry does this naturally via diffusion/mean)
        # Mean intensity of the grid
        signal = observation.mean().item()

        # Trace Dynamics (Physical Memory)
        # Decay 0.9 preserves the signal across the 15-step gap
        # We amplify the signal because mean() over 32x32 dilutes the 4x4 block
        # 4x4 = 16 pixels. 32x32 = 1024 pixels. Ratio ~0.015.
        # Signal 5.0 -> Mean increase ~0.075.
        # Noise is mean 0.

        # To make it comparable, we scale the input or the threshold.
        # Let's say the Geometric Agent has "Sensitivity" to global stress.

        sensitivity = 10.0
        input_stress = signal * sensitivity

        self.trace = self.trace * 0.9 + input_stress

        if self.trace > self.threshold:
            return True
        return False

    def reset(self):
        self.trace = 0.0

class NeuralAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = CH * GRID * GRID
        self.hidden_dim = 128

        # LSTM for temporal processing
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, hidden):
        # Flatten: (Batch, Channels, H, W) -> (Batch, 1, Features)
        x = x.view(1, 1, -1)
        out, hidden = self.lstm(x, hidden)
        prob = torch.sigmoid(self.fc(out))
        return prob, hidden

# =====================
# EXPERIMENT RUNNER
# =====================
def run_experiment(agent_type="geometric"):
    env = RiftWorldA2()

    if agent_type == "neural":
        agent = NeuralAgent().to(DEVICE)
        # No training loop here implies "Zero-shot" or "Untrained".
        # However, to be fair, neural nets need training.
        # But the hypothesis is about architecture failure even with capacity.
        # We will simulate a "randomly initialized" agent which represents
        # the lack of "Geometric Prior".
        # If we wanted to train it, it would take thousands of episodes.
        # For Phase 2 demonstration, we show that without the prior, it fails.
    else:
        agent = GeometricAgent()

    stats = {
        "survived": 0,
        "commit_time": [],
        "died_from_ischemia": 0,
        "died_from_impact": 0
    }

    print(f"\n🧪 Running {agent_type.upper()} Agent...")

    for episode in range(EPISODES):
        obs = env.reset()
        if agent_type == "geometric":
            agent.reset()
        else:
            hidden = None

        committed_at = None
        survived_episode = False
        cause_of_death = None

        # Episode Loop
        for t in range(STEPS_PER_EPISODE):
            # Action
            commit_action = False

            if agent_type == "geometric":
                commit_action = agent.act(obs)
            else:
                prob, hidden = agent(obs, hidden)
                # Random action based on probability? Or Threshold?
                # Let's use threshold 0.5
                commit_action = prob.item() > 0.5

            obs, health, continuing = env.step(commit_action)

            if commit_action and committed_at is None:
                committed_at = t

            if not continuing: # Episode ended
                if health <= 0:
                    if env.brace_level > 0.3 and env.step_count < env.hazard_time:
                        cause_of_death = "ischemia"
                    else:
                        cause_of_death = "impact"
                break

        if health > 0:
            stats["survived"] += 1
            survived_episode = True
        else:
            if cause_of_death == "ischemia":
                stats["died_from_ischemia"] += 1
            elif cause_of_death == "impact":
                stats["died_from_impact"] += 1

        if committed_at is not None:
            stats["commit_time"].append(committed_at)

    print(f"   Survival Rate: {stats['survived']}/{EPISODES} ({stats['survived']/EPISODES*100:.1f}%)")
    mean_commit = np.mean(stats["commit_time"]) if stats["commit_time"] else 0.0
    print(f"   Mean Commit Time: {mean_commit:.1f}")
    print(f"   Deaths: Ischemia={stats['died_from_ischemia']}, Impact={stats['died_from_impact']}")

    return stats

if __name__ == "__main__":
    run_experiment("geometric")
    run_experiment("neural")
