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
STEPS_TRAIN = 300  # Fewer episodes needed if the signal is strong
STEPS_PER_EPISODE = 60

print(f"🦁 PROJECT CHIMERA v2: CATASTROPHIC PENALTY ON {DEVICE}")
print("   (Hypothesis: Increasing the cost of failure will force the Brain to use the Gate.)")

# =====================
# THE BODY (Multi-Scale Memory)
# =====================
class RiftBody(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0.5, 1.0, 0.5], 
                              [1.0, -6.0, 1.0], 
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        
        # MEMORY: Fast Trace (Reflex) & Slow Trace (Context)
        self.trace_fast = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.trace_slow = torch.zeros(1, CH, GRID, GRID, device=DEVICE)

    def reset(self):
        # Detach from graph
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.trace_fast = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.trace_slow = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        return self.get_sensor_data()

    def step(self, external_force, cue, gate_action):
        # 1. Update Traces (The Body remembers the Cue)
        self.trace_fast = self.trace_fast * 0.8 + cue * 1.0
        self.trace_slow = self.trace_slow * 0.98 + cue * 0.5
        
        # 2. Gated Physics
        # Gate: 0 (Closed) to 1 (Open)
        gated_force = external_force * gate_action

        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -1.0 * self.u
        accel = diffusion + restoring + gated_force - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 3. CALCULATE PAIN (The Death Penalty)
        deformation = self.u.abs().max() # Peak damage
        
        # CATASTROPHIC PENALTY:
        # If deformation exceeds safe limits, pain creates a massive spike.
        # This prevents the "lazy optimization" loop.
        if deformation > 0.5:
            pain_signal = deformation * 100.0 # DEATH
        else:
            pain_signal = deformation * 1.0   # Discomfort
            
        gate_cost = (1.0 - gate_action) * 0.01 # Cheap to close eyes
        
        total_pain = pain_signal + gate_cost
        
        return total_pain, self.get_sensor_data()

    def get_sensor_data(self):
        # Brain sees: [Vibration, FastTrace, SlowTrace]
        return torch.stack([
            self.u.abs().mean(),
            self.trace_fast.mean(),
            self.trace_slow.mean()
        ]).unsqueeze(0)

# =====================
# THE BRAIN
# =====================
class ChimeraBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=32, batch_first=True)
        self.fc_action = nn.Linear(32, 1)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        # Action -> 0 to 1
        action = torch.sigmoid(self.fc_action(out))
        return action, hidden

# =====================
# RUNNER
# =====================
def run():
    body = RiftBody().to(DEVICE)
    brain = ChimeraBrain().to(DEVICE)
    optimizer = optim.Adam(brain.parameters(), lr=0.005) # Lower LR for stability
    
    print("\n🧠 TRAINING: Survival Mode")
    
    for episode in range(STEPS_TRAIN):
        body.reset()
        hidden = None
        total_loss = 0
        
        gate_history = []
        
        for t in range(STEPS_PER_EPISODE):
            # INPUTS
            cue = torch.zeros_like(body.u)
            if t == 10: cue[:,:,16-4:16+4,16-4:16+4] = 1.0 # Cue
            
            force = torch.zeros_like(body.u)
            if t == 25: force[:,:,16-2:16+2,16-2:16+2] = 5.0 # Hammer
            
            # SENSE
            state = body.get_sensor_data().unsqueeze(1)
            
            # THINK
            gate_action_tensor, hidden = brain(state, hidden)
            gate_action = gate_action_tensor.squeeze()
            gate_history.append(gate_action.item())
            
            # ACT
            pain, _ = body.step(force, cue, gate_action)
            total_loss += pain
            
        # LEARN
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0) # Stabilize gradients
        optimizer.step()
        
        if episode % 50 == 0:
            print(f"   Episode {episode:03d} | Total Pain: {total_loss.item():.4f} | Gate @ Impact: {gate_history[25]:.4f}")

    print("\n🧪 FINAL TEST: Cued vs Uncued")
    
    # TEST 1: UNCUED
    body.reset()
    hidden = None
    pain_uncued = 0
    gate_uncued = []
    for t in range(STEPS_PER_EPISODE):
        state = body.get_sensor_data().unsqueeze(1)
        with torch.no_grad():
            gate, hidden = brain(state, hidden)
        gate_uncued.append(gate.item())
        
        force = torch.zeros_like(body.u)
        if t == 25: force[:,:,16-2:16+2,16-2:16+2] = 5.0
        p, _ = body.step(force, torch.zeros_like(body.u), gate.squeeze())
        pain_uncued += p.item()

    # TEST 2: CUED
    body.reset()
    hidden = None
    pain_cued = 0
    gate_cued = []
    for t in range(STEPS_PER_EPISODE):
        cue = torch.zeros_like(body.u)
        if t == 10: cue[:,:,16-4:16+4,16-4:16+4] = 1.0
        
        state = body.get_sensor_data().unsqueeze(1)
        with torch.no_grad():
            gate, hidden = brain(state, hidden)
        gate_cued.append(gate.item())
        
        force = torch.zeros_like(body.u)
        if t == 25: force[:,:,16-2:16+2,16-2:16+2] = 5.0
        p, _ = body.step(force, cue, gate.squeeze())
        pain_cued += p.item()

    print("\n📊 CHIMERA v2 REPORT")
    print(f"   Pain (Uncued): {pain_uncued:.2f}")
    print(f"   Pain (Cued):   {pain_cued:.2f}")
    print(f"   Gate @ Impact (Uncued): {gate_uncued[25]:.4f}")
    print(f"   Gate @ Impact (Cued):   {gate_cued[25]:.4f}")
    
    if pain_cued < pain_uncued * 0.5:
        print("   ✅ SUCCESS: System learned to close the gate to survive.")
    else:
        print("   ❌ FAILURE: Optimization failed.")

if __name__ == "__main__":
    run()