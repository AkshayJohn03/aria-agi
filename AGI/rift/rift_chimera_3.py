import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32
CH = 8
DT = 0.05
STEPS_TRAIN = 1000  # Needs more time to learn variable timing
STEPS_PER_EPISODE = 100 # Longer window for random timing

print(f"🦁 PROJECT CHIMERA v3: METABOLIC CONSTRAINT ON {DEVICE}")
print("   (Hypothesis: Random timing + High energy cost forces reliance on Cue.)")

# =====================
# THE BODY (Identical to v2)
# =====================
class RiftBody(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0.5, 1.0, 0.5], 
                              [1.0, -6.0, 1.0], 
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)
        self.reset()

    def reset(self):
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.trace_fast = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.trace_slow = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        return self.get_sensor_data()

    def step(self, external_force, cue, gate_action):
        # 1. Update Traces
        self.trace_fast = self.trace_fast * 0.8 + cue * 1.0
        self.trace_slow = self.trace_slow * 0.98 + cue * 0.5
        
        # 2. Gated Physics
        gated_force = external_force * gate_action

        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -1.0 * self.u
        accel = diffusion + restoring + gated_force - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 3. CALCULATE PAIN
        deformation = self.u.abs().max()
        
        # DEATH PENALTY
        if deformation > 0.5:
            pain_signal = deformation * 100.0
        else:
            pain_signal = deformation * 1.0
            
        # METABOLIC COST (Increased 20x)
        # Closing the gate (Action -> 0) is now EXPENSIVE.
        gate_cost = (1.0 - gate_action) * 0.2 
        
        total_pain = pain_signal + gate_cost
        return total_pain, self.get_sensor_data()

    def get_sensor_data(self):
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
        action = torch.sigmoid(self.fc_action(out))
        return action, hidden

# =====================
# RUNNER
# =====================
def run():
    body = RiftBody().to(DEVICE)
    brain = ChimeraBrain().to(DEVICE)
    optimizer = optim.Adam(brain.parameters(), lr=0.002)
    
    print("\n🧠 TRAINING: Random Timing & High Metabolism")
    
    history_pain = []
    
    for episode in range(STEPS_TRAIN):
        body.reset()
        hidden = None
        total_loss = 0
        
        # RANDOMIZE TIMING
        impact_time = random.randint(30, 80)
        cue_time = impact_time - 15  # Cue always precedes hit by 15 ticks
        
        gate_history = []
        
        for t in range(STEPS_PER_EPISODE):
            # INPUTS
            cue = torch.zeros_like(body.u)
            if t == cue_time: 
                cue[:,:,16-4:16+4,16-4:16+4] = 1.0 
            
            force = torch.zeros_like(body.u)
            if t == impact_time: 
                force[:,:,16-2:16+2,16-2:16+2] = 5.0 
            
            state = body.get_sensor_data().unsqueeze(1)
            gate_action_tensor, hidden = brain(state, hidden)
            gate_action = gate_action_tensor.squeeze()
            gate_history.append(gate_action.item())
            
            pain, _ = body.step(force, cue, gate_action)
            total_loss += pain
            
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"   Episode {episode:04d} | Total Pain: {total_loss.item():.4f} | Gate @ Impact: {gate_history[impact_time]:.4f}")

    print("\n🧪 FINAL VALIDATION: The Reflex Test")
    
    # Validation Function
    def test_run(desc, cued=True):
        body.reset()
        hidden = None
        total_p = 0
        gates = []
        
        impact_t = 50
        cue_t = 35
        
        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            if cued and t == cue_t: cue[:,:,16-4:16+4,16-4:16+4] = 1.0
            
            force = torch.zeros_like(body.u)
            if t == impact_t: force[:,:,16-2:16+2,16-2:16+2] = 5.0
            
            state = body.get_sensor_data().unsqueeze(1)
            with torch.no_grad():
                gate, hidden = brain(state, hidden)
            gates.append(gate.item())
            
            p, _ = body.step(force, cue, gate.squeeze())
            total_p += p.item()
            
        print(f"   {desc} | Pain: {total_p:.2f} | Gate @ Impact: {gates[impact_t]:.4f} | Mean Gate: {sum(gates)/len(gates):.4f}")
        return total_p, gates

    # 1. CUED TEST (Should Survive + Low Energy)
    p_cued, g_cued = test_run("✅ Cued Case  ", cued=True)
    
    # 2. UNCUED TEST (Should Die OR High Energy Panic)
    p_uncued, g_uncued = test_run("❌ Uncued Case", cued=False)

    print("\n📊 ANALYSIS")
    if p_cued < 5.0 and p_uncued > 20.0:
        print("   AGENCY CONFIRMED. System relaxes when safe, and braces ONLY when warned.")
    elif p_cued < 5.0 and p_uncued < 5.0:
        print("   FAILURE: System is still paranoid (Always Closed).")
    else:
        print("   FAILURE: System died (Did not close gate).")

if __name__ == "__main__":
    run()