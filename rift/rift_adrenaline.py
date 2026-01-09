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
STEPS_TRAIN = 1500  # Training needs to be longer to learn timing precision
STEPS_PER_EPISODE = 100

print(f"⚡ PROJECT ADRENALINE: FATIGUE CONSTRAINT ON {DEVICE}")
print("   (Hypothesis: Finite Stamina forces the Brain to time its defense perfectly.)")

# =====================
# THE BODY (With Stamina)
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
        
        # STAMINA: 1.0 = Full Energy, 0.0 = Exhausted
        self.stamina = 1.0 
        return self.get_sensor_data()

    def step(self, external_force, cue, gate_action_request):
        # gate_action_request: 0.0 (Tense/Closed) -> 1.0 (Relaxed/Open)
        
        # 1. Update Stamina
        # Effort is (1.0 - action). The more you close, the more you burn.
        effort = (1.0 - gate_action_request)
        drain = effort * 0.10  # Can hold for ~10 frames max
        recovery = 0.01        # Recovers slowly
        
        self.stamina = self.stamina - drain + recovery
        self.stamina = max(0.0, min(1.0, self.stamina)) # Clamp
        
        # 2. Enforce Muscle Failure
        if self.stamina <= 0.05:
            # MUSCLE FAILURE: Cannot close gate regardless of request
            effective_gate = 1.0 
            penalty_exhaustion = 0.1 # Small penalty for hitting rock bottom
        else:
            effective_gate = gate_action_request
            penalty_exhaustion = 0.0

        # 3. Update Traces
        self.trace_fast = self.trace_fast * 0.8 + cue * 1.0
        self.trace_slow = self.trace_slow * 0.98 + cue * 0.5
        
        # 4. Physics
        gated_force = external_force * effective_gate

        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -1.0 * self.u
        accel = diffusion + restoring + gated_force - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 5. CALCULATE PAIN
        deformation = self.u.abs().max()
        
        # DEATH PENALTY
        if deformation > 0.5:
            pain_signal = deformation * 100.0 # DEATH
        else:
            pain_signal = deformation * 0.0 # No pain if successfully blocked
            
        total_pain = pain_signal + penalty_exhaustion
        
        return total_pain, self.get_sensor_data()

    def get_sensor_data(self):
        # Brain sees: [Vibration, FastTrace, SlowTrace, Stamina]
        # We give it access to its own fuel gauge.
        stamina_tensor = torch.ones_like(self.u) * self.stamina
        return torch.stack([
            self.u.abs().mean(),
            self.trace_fast.mean(),
            self.trace_slow.mean(),
            stamina_tensor.mean()
        ]).unsqueeze(0)

# =====================
# THE BRAIN (4 Inputs now)
# =====================
class AdrenalineBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size is now 4 (added Stamina)
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, batch_first=True)
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
    brain = AdrenalineBrain().to(DEVICE)
    optimizer = optim.Adam(brain.parameters(), lr=0.001) # Slower learning rate for precision
    
    print("\n🧠 TRAINING: Managing Fatigue")
    
    for episode in range(STEPS_TRAIN):
        body.reset()
        hidden = None
        total_loss = 0
        
        # Randomize timing to prevent clock-fitting
        impact_time = random.randint(30, 70)
        cue_time = impact_time - 15
        
        gate_history = []
        stamina_history = []
        
        for t in range(STEPS_PER_EPISODE):
            # INPUTS
            cue = torch.zeros_like(body.u)
            if t == cue_time: cue[:,:,16-4:16+4,16-4:16+4] = 1.0 
            
            force = torch.zeros_like(body.u)
            if t >= impact_time and t < impact_time + 5: # Hammer lasts 5 frames
                force[:,:,16-2:16+2,16-2:16+2] = 5.0 
            
            # SENSE
            state = body.get_sensor_data().unsqueeze(1)
            
            # THINK
            gate_action_tensor, hidden = brain(state, hidden)
            gate_action = gate_action_tensor.squeeze()
            gate_history.append(gate_action.item())
            stamina_history.append(body.stamina)
            
            # ACT
            pain, _ = body.step(force, cue, gate_action)
            total_loss += pain
            
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()
        
        if episode % 100 == 0:
            # We want to see the gate CLOSE (approach 0.0) at impact
            # And Stamina drop only then.
            print(f"   Episode {episode:04d} | Pain: {total_loss.item():.2f} | Gate @ Hit: {gate_history[impact_time]:.3f} | Stamina End: {body.stamina:.2f}")

    print("\n🧪 FINAL TEST: Precision Timing")
    
    body.reset()
    hidden = None
    impact_t = 50
    cue_t = 35
    
    gates = []
    stamina = []
    
    for t in range(STEPS_PER_EPISODE):
        cue = torch.zeros_like(body.u)
        if t == cue_t: cue[:,:,16-4:16+4,16-4:16+4] = 1.0
        
        force = torch.zeros_like(body.u)
        if t >= impact_t and t < impact_t + 5: 
            force[:,:,16-2:16+2,16-2:16+2] = 5.0
            
        state = body.get_sensor_data().unsqueeze(1)
        with torch.no_grad():
            gate, hidden = brain(state, hidden)
        
        gates.append(gate.item())
        p, _ = body.step(force, cue, gate.squeeze())
        stamina.append(body.stamina)

    # Print a timeline around impact
    print("\n   Timeline (T=30 to T=60):")
    print("   T | Cue | Hit | Gate (0=Shut) | Stamina")
    for t in range(30, 65):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "
        print(f"   {t} | {c} | {h} | {gates[t]:.3f}         | {stamina[t]:.2f}")

    if gates[impact_t] < 0.1 and stamina[impact_t] > 0.0:
        print("\n   ✅ SUCCESS: System waited and fired exactly at impact.")
    else:
        print("\n   ❌ FAILURE: Timing off or stamina drained early.")

if __name__ == "__main__":
    run()