import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32
CH = 8
DT = 0.05
STEPS_TRAIN = 500  # Episodes
STEPS_PER_EPISODE = 60

print(f"🦁 PROJECT CHIMERA: HYBRID NEURAL-PHYSICS AGENT ON {DEVICE}")
print("   (Hypothesis: A neural controller can bridge the temporal gap that physics missed.)")

# =====================
# THE BODY (RIFT Substrate)
# =====================
class RiftBody(nn.Module):
    def __init__(self):
        super().__init__()
        # Physics Kernels
        lap = torch.tensor([[[[0.5, 1.0, 0.5],
                              [1.0, -6.0, 1.0],
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        # State
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.k = 1.0 # Base stiffness

    def reset(self):
        # DETACH: Create fresh tensors so we don't backprop into the previous episode
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        return self.get_sensor_data()

    def step(self, external_force, gate_action):
        # 1. Gating (Action from Brain)
        # Gate is 0.0 (Closed) to 1.0 (Open)
        # The Brain controls how much force enters the body
        gated_force = external_force * gate_action

        # 2. Physics (The Body reacts)
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -self.k * self.u
        accel = diffusion + restoring + gated_force - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 3. Pain (Reward Signal)
        # Pain = Deformation + Cost of Closing Gate (Metabolic tax)
        # We want the brain to only close the gate when necessary.
        deformation = self.u.abs().mean()
        gate_cost = (1.0 - gate_action) * 0.02

        total_pain = deformation + gate_cost

        return total_pain, self.get_sensor_data()

    def get_sensor_data(self):
        # The "Nervous System" summary sent to the Brain
        # [Mean Vibration, Max Vibration]
        return torch.stack([
            self.u.abs().mean(),
            self.u.abs().max()
        ]).unsqueeze(0) # Shape (1, 2)

# =====================
# THE BRAIN (Neural Controller)
# =====================
class ChimeraBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 2 (Body Sensors) + 1 (Visual Cue)
        self.input_dim = 3
        self.hidden_dim = 16

        # LSTM allows Short-Term Memory (bridging the time gap)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)

        # Output: Gate Value (Sigmoid 0-1)
        self.fc_action = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, hidden):
        # x shape: (Batch, Seq, Features)
        out, hidden = self.lstm(x, hidden)
        action = torch.sigmoid(self.fc_action(out))
        return action, hidden

# =====================
# RUNNER
# =====================
def run():
    body = RiftBody().to(DEVICE)
    brain = ChimeraBrain().to(DEVICE)
    optimizer = optim.Adam(brain.parameters(), lr=0.01)

    print("\n🧠 TRAINING PHASE: Teaching the Chimera to 'Duck'")

    history_pain = []

    for episode in range(STEPS_TRAIN):
        body.reset()

        # Reset Brain Memory
        hidden = None

        total_episode_loss = 0
        total_episode_pain = 0

        # We need to collect states to backpropagate
        # Simple policy gradient / reinforce style is hard,
        # let's try direct differentiable physics guidance since Pytorch allows it!

        outputs = []

        # TIMELINE:
        # T=10: Cue (Visual)
        # T=25: Impact (Physical)

        for t in range(STEPS_PER_EPISODE):
            # 1. OBSERVATION
            # Visual Cue (External Sensor)
            visual_cue = 1.0 if t == 10 else 0.0
            visual_tensor = torch.tensor([[visual_cue]], device=DEVICE)

            # Body Sensation
            body_sense = body.get_sensor_data() # (1, 2)

            # Fuse Input: [MeanVib, MaxVib, VisualCue]
            brain_input = torch.cat([body_sense, visual_tensor], dim=1).unsqueeze(1) # (1, 1, 3)

            # 2. COGNITION
            gate_action_tensor, hidden = brain(brain_input, hidden)
            gate_action = gate_action_tensor.squeeze()

            # 3. PHYSICS INTERACTION
            force = torch.zeros_like(body.u)
            if t == 25:
                # The Hammer
                force[:,:,16-2:16+2,16-2:16+2] = 5.0

            pain, _ = body.step(force, gate_action)

            # Accumulate Loss
            # We want to minimize Pain.
            total_episode_loss += pain
            total_episode_pain += pain.item()

        # 4. LEARNING (Backprop through Time & Physics)
        optimizer.zero_grad()
        total_episode_loss.backward()
        optimizer.step()

        history_pain.append(total_episode_pain)

        if episode % 50 == 0:
            print(f"   Episode {episode:03d} | Total Pain: {total_episode_pain:.4f}")

    print("\n🧪 TEST PHASE: Cued vs Uncued")

    # TEST 1: UNCUED (Brain sees no warning)
    body.reset()
    hidden = None
    pain_uncued = 0
    print("   Running Uncued (Unexpected)...")
    for t in range(STEPS_PER_EPISODE):
        brain_input = torch.cat([body.get_sensor_data(), torch.zeros((1,1), device=DEVICE)], dim=1).unsqueeze(1)
        with torch.no_grad():
            gate_action, hidden = brain(brain_input, hidden)

        force = torch.zeros_like(body.u)
        if t == 25: force[:,:,16-2:16+2,16-2:16+2] = 5.0

        p, _ = body.step(force, gate_action.squeeze())
        pain_uncued += p.item()

    # TEST 2: CUED (Brain sees warning)
    body.reset()
    hidden = None
    pain_cued = 0
    gate_trace = []

    print("   Running Cued (Expected)...")
    for t in range(STEPS_PER_EPISODE):
        visual_cue = 1.0 if t == 10 else 0.0
        brain_input = torch.cat([body.get_sensor_data(), torch.tensor([[visual_cue]], device=DEVICE)], dim=1).unsqueeze(1)

        with torch.no_grad():
            gate_action, hidden = brain(brain_input, hidden)

        gate_trace.append(gate_action.item())

        force = torch.zeros_like(body.u)
        if t == 25: force[:,:,16-2:16+2,16-2:16+2] = 5.0

        p, _ = body.step(force, gate_action.squeeze())
        pain_cued += p.item()

    print("\n📊 CHIMERA REPORT")
    print(f"   Pain (Uncued): {pain_uncued:.4f}")
    print(f"   Pain (Cued):   {pain_cued:.4f}")

    delta = pain_uncued - pain_cued
    pct = (delta / pain_uncued) * 100 if pain_uncued > 0 else 0

    print(f"   Pain Reduction: {pct:.2f}%")

    # Check Gate Timing
    gate_at_impact = gate_trace[25]
    print(f"   Gate at Impact (T=25): {gate_at_impact:.4f} (Should be low/closed)")

    if pct > 50.0:
        print("   ✅ SUCCESS: The Chimera learned to close its eyes before the punch.")
    else:
        print("   ❌ FAILURE: Optimization failed.")

if __name__ == "__main__":
    run()