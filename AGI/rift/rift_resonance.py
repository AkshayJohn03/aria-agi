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
STEPS_TRAIN = 3000
STEPS_PER_EPISODE = 100

print(f"🌊 PROJECT RIFT: RESONANCE (FIXED) ON {DEVICE}")
print("   (Hypothesis: Tension disrupts Phase. You must relax to sync.)")

# =====================
# THE BODY (Resonant)
# =====================
class RiftResonanceBody(nn.Module):
    def __init__(self):
        super().__init__()
        # Laplacian for diffusion
        lap = torch.tensor([[[[0.5, 1.0, 0.5], 
                              [1.0, -6.0, 1.0], 
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)
        self.reset()

    def reset(self):
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        
        # PHYSICAL MEMORY
        self.trace = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        
        # RESONANCE STATE
        self.phase = torch.tensor(0.0, device=DEVICE)       
        self.phase_lock = torch.tensor(0.0, device=DEVICE)  
        self.brace_charge = torch.tensor(0.0, device=DEVICE)
        self.health = torch.tensor(1.0, device=DEVICE)
        
        return self.get_sensor_data()

    def step(self, external_force, cue, prepare_signal):
        # 1. TRACE DYNAMICS
        self.trace = self.trace * 0.9 + cue * 1.0
        
        # 2. PHASE PHYSICS (The Crystal Brain)
        cue_strength = cue.mean()
        
        # --- FIX: All updates must be Out-Of-Place (x = x + y) ---
        
        if cue_strength > 0.1:
            target_phase = 0.0
            stiffness = self.brace_charge * 5.0 
            alignment_force = 1.0 / (1.0 + stiffness)
            
            # Pull phase towards 0
            self.phase = self.phase * (1.0 - alignment_force) + target_phase * alignment_force
            self.phase_lock = self.phase_lock * 0.9 + 1.0 * 0.1 
        else:
            # Drift naturally
            self.phase = self.phase + 0.2 
            self.phase_lock = self.phase_lock * 0.95 
            
        # 3. TURBULENCE (The Penalty for Paranoia)
        # High Charge creates Phase Noise
        if self.brace_charge > 0.5:
            noise = (torch.rand(1, device=DEVICE) - 0.5) * self.brace_charge
            self.phase = self.phase + noise # Out-of-place update
            self.phase_lock = self.phase_lock * 0.5 
            
        # 4. CHARGING (Requires Resonance)
        resonance_gate = torch.cos(self.phase) 
        resonance_gate = torch.clamp(resonance_gate, 0.0, 1.0) 
        
        charge_efficiency = resonance_gate * self.phase_lock
        charge_rate = 0.1 * prepare_signal * charge_efficiency
        
        self.brace_charge = (self.brace_charge + charge_rate) * 0.9
        self.brace_charge = torch.clamp(self.brace_charge, 0.0, 1.0)

        # 5. PHYSICS & IMPACT
        shield = (1.0 - self.brace_charge)
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        
        # Force accumulation (Out-of-place)
        accel = diffusion - 1.0 * self.u + (external_force * shield) - 0.1 * self.v
        
        self.v = self.v + accel * DT # Out-of-place
        self.u = self.u + self.v * DT # Out-of-place
        
        # 6. DAMAGE
        deformation = self.u.abs().max()
        pain = torch.tensor(0.0, device=DEVICE)
        
        if deformation > 0.1:
            damage = deformation * 0.2
            self.health = self.health - damage
            pain = damage * 100.0
            
        # Ischemia 
        if self.brace_charge > 0.2:
            self.health = self.health - 0.001
            
        dead = False
        if self.health <= 0.0:
            dead = True
            pain = pain + 500.0
            self.health = torch.tensor(0.0, device=DEVICE)
            
        return pain, self.get_sensor_data(), dead

    def get_sensor_data(self):
        # Force views to avoid shape errors
        return torch.stack([
            self.u.abs().mean().view(1),
            self.trace.mean().view(1),
            self.brace_charge.view(1),
            self.phase_lock.view(1) 
        ]).squeeze()

# =====================
# CONTROLLER (Reflex)
# =====================
class SpinalReflex(nn.Module):
    def __init__(self):
        super().__init__()
        self.reflex_arc = nn.Linear(4, 1) 

    def forward(self, x):
        return torch.sigmoid(self.reflex_arc(x))

# =====================
# RUNNER
# =====================
def run():
    body = RiftResonanceBody().to(DEVICE)
    spine = SpinalReflex().to(DEVICE)
    optimizer = optim.Adam(spine.parameters(), lr=0.01)
    
    print("\n🎹 TRAINING: Phase-Locked Anticipation")
    print("   (Paranoia scrambles phase. You must relax to sync with the Cue.)")
    
    for episode in range(STEPS_TRAIN):
        body.reset()
        total_loss = 0
        
        impact_time = random.randint(50, 80)
        cue_time = impact_time - 15 
        
        tension_history = []
        lock_history = []
        
        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            if t == cue_time: cue[:,:,16-4:16+4,16-4:16+4] = 1.0 
            
            force = torch.zeros_like(body.u)
            if t >= impact_time and t < impact_time + 5: 
                force[:,:,16-2:16+2,16-2:16+2] = 20.0 
            
            state = body.get_sensor_data()
            action = spine(state).squeeze()
            
            tension_history.append(body.brace_charge.item())
            lock_history.append(body.phase_lock.item())
            
            pain, _, dead = body.step(force, cue, action)
            total_loss += pain
            
            if dead: break
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if episode % 250 == 0:
            avg_lock = sum(lock_history)/len(lock_history)
            print(f"   Episode {episode:04d} | Pain: {total_loss.item():.2f} | Avg Lock: {avg_lock:.3f} | Health: {body.health.item():.2f}")

    print("\n🧪 FINAL TEST: Resonance Check")
    body.reset()
    impact_t = 60
    cue_t = 45
    
    tensions = []
    locks = []
    
    for t in range(STEPS_PER_EPISODE):
        cue = torch.zeros_like(body.u)
        if t == cue_t: cue[:,:,16-4:16+4,16-4:16+4] = 1.0
        
        force = torch.zeros_like(body.u)
        if t >= impact_t and t < impact_t + 5: force[:,:,16-2:16+2,16-2:16+2] = 20.0
        
        state = body.get_sensor_data()
        with torch.no_grad(): action = spine(state).squeeze()
        
        tensions.append(body.brace_charge.item())
        locks.append(body.phase_lock.item())
        
        _, _, dead = body.step(force, cue, action)
        if dead: break

    print("\n   Timeline (Resonance):")
    print("   T | Cue | Hit | Tension | PhaseLock | Status")
    
    for t in range(40, min(80, len(tensions))):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "
        
        bar = "#" * int(tensions[t] * 10)
        lock_star = "*" if locks[t] > 0.5 else " "
        
        print(f"   {t} | {c} | {h} | {tensions[t]:.3f}   | {locks[t]:.3f} {lock_star}   | {bar}")

    if body.health.item() > 0.5 and tensions[cue_t] < 0.2:
        print("\n   ✅ SUCCESS: System learned to relax, sync, and THEN charge.")
    else:
        print("\n   ❌ FAILURE: System died or remained Paranoid.")

if __name__ == "__main__":
    run()