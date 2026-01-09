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
STEPS_TRAIN = 2500 # More steps needed to find the delicate balance
STEPS_PER_EPISODE = 100

print(f"🩸 PROJECT RIFT: ISCHEMIA (FATIGUE-OF-READINESS) ON {DEVICE}")
print("   (Hypothesis: Sustained tension causes necrosis. Timing is now mandatory.)")

# =====================
# THE BODY (Ischemic)
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
        
        self.brace_charge = torch.tensor(0.0, device=DEVICE)
        self.health = torch.tensor(1.0, device=DEVICE) 
        
        return self.get_sensor_data()

    def step(self, external_force, cue, prepare_signal):
        # 1. TRACES
        self.trace_fast = self.trace_fast * 0.8 + cue * 1.0
        self.trace_slow = self.trace_slow * 0.98 + cue * 0.5
        
        # 2. MUSCLE DYNAMICS
        # Standard charge/leak dynamics
        # Note: We reduced leak slightly to make holding easier, 
        # so the constraint is purely Ischemia, not just leak cost.
        effective_rate = 0.05 * self.health 
        charge_rate = effective_rate * prepare_signal
        leak_rate = 0.99 
        
        self.brace_charge = (self.brace_charge + charge_rate) * leak_rate
        self.brace_charge = torch.clamp(self.brace_charge, 0.0, 1.0)
        
        # 3. ISCHEMIC NECROSIS (The New Constraint)
        # If you hold tension > 0.2, you take damage.
        # Rate 0.02 means 50 steps of tension = DEATH.
        ischemic_damage = torch.tensor(0.0, device=DEVICE)
        if self.brace_charge > 0.2:
            ischemic_damage = self.brace_charge * 0.02
            self.health = self.health - ischemic_damage

        # 4. CONTINUOUS SHIELD
        shield_factor = (1.0 - self.brace_charge)
        dampened_force = external_force * shield_factor

        # 5. PHYSICS
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -1.0 * self.u
        accel = diffusion + restoring + dampened_force - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 6. TRAUMATIC DAMAGE
        deformation = self.u.abs().max()
        pain_signal = torch.tensor(0.0, device=DEVICE)
        
        if deformation > 0.1:
            traumatic_damage = deformation * 0.1 
            self.health = self.health - traumatic_damage
            pain_signal = traumatic_damage * 1000.0 
        
        # 7. MORTALITY CHECK
        dead = False
        if self.health <= 0.0:
            dead = True
            pain_signal += 1000.0 
            self.health = torch.tensor(0.0, device=DEVICE)

        # 8. COST FUNCTION
        # We add ischemic pain so the brain feels the burn of holding on.
        pain_signal += ischemic_damage * 100.0
        
        return pain_signal, self.get_sensor_data(), dead

    def get_sensor_data(self):
        return torch.stack([
            self.u.abs().mean().view(1),
            self.trace_fast.mean().view(1),
            self.brace_charge.view(1),
            self.health.view(1)
        ]).squeeze()

# =====================
# THE SPINAL CORD
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
    body = RiftBody().to(DEVICE)
    spine = SpinalReflex().to(DEVICE)
    optimizer = optim.Adam(spine.parameters(), lr=0.01)
    
    print("\n🧠 TUNING: Balancing Trauma vs. Ischemia")
    
    for episode in range(STEPS_TRAIN):
        body.reset()
        total_loss = 0
        
        impact_time = random.randint(40, 80)
        cue_time = impact_time - 15
        
        tension_history = []
        health_history = []
        
        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            if t == cue_time: cue[:,:,16-4:16+4,16-4:16+4] = 1.0 
            
            force = torch.zeros_like(body.u)
            if t >= impact_time and t < impact_time + 5: 
                force[:,:,16-2:16+2,16-2:16+2] = 20.0 
            
            sensor_state = body.get_sensor_data()
            prepare_signal = spine(sensor_state).squeeze()
            
            tension_history.append(body.brace_charge.item())
            health_history.append(body.health.item())
            
            pain, _, dead = body.step(force, cue, prepare_signal)
            total_loss += pain
            
            if dead:
                break
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if episode % 250 == 0:
            if len(tension_history) > impact_time:
                tens = tension_history[impact_time-1]
                hlth = health_history[impact_time]
            else:
                tens = tension_history[-1] 
                hlth = 0.0
                
            print(f"   Episode {episode:04d} | Loss: {total_loss.item():.2f} | Tension: {tens:.3f} | Health: {hlth:.2f}")

    print("\n🧪 FINAL TEST: Anticipation Check")
    
    body.reset()
    impact_t = 60
    cue_t = 45
    
    tensions = []
    healths = []
    
    for t in range(STEPS_PER_EPISODE):
        cue = torch.zeros_like(body.u)
        if t == cue_t: cue[:,:,16-4:16+4,16-4:16+4] = 1.0
        
        force = torch.zeros_like(body.u)
        if t >= impact_t and t < impact_t + 5: 
            force[:,:,16-2:16+2,16-2:16+2] = 20.0
        
        state = body.get_sensor_data()
        with torch.no_grad():
            action = spine(state).squeeze()
            
        tensions.append(body.brace_charge.item())
        healths.append(body.health.item())
        
        pain, _, dead = body.step(force, cue, action)
        
        if dead:
            print(f"   💀 DIED at step {t}")
            break

    print("\n   Timeline (Ischemic Constraint):")
    print("   T | Cue | Hit | Charge | Health")
    limit = min(80, len(tensions))
    for t in range(40, limit):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "
        
        # Visualize Charge
        bar = "#" * int(tensions[t] * 10)
        
        print(f"   {t} | {c} | {h} | {tensions[t]:.3f}  | {healths[t]:.2f} {bar}")

    survived = len(healths) > impact_t + 5 and healths[-1] > 0.0
    relaxed_before_cue = sum(tensions[40:cue_t]) / (cue_t - 40) < 0.3
    
    if survived and relaxed_before_cue:
        print("\n   ✅ SUCCESS: TRUE ANTICIPATION. System relaxes, waits, braces, and survives.")
    elif survived:
        print("\n   ⚠️ PARTIAL: Survived, but maintained high tension (Paranoia).")
    else:
        print("\n   ❌ FAILURE: Died (Trauma or Ischemia).")

if __name__ == "__main__":
    run()