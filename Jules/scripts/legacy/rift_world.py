import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32
CH = 8
DT = 0.05
STEPS_TRAIN = 3000  # Long training to find the needle in the haystack
STEPS_PER_EPISODE = 100

print(f"🌍 RIFT-WORLD: UNIFIED PRESSURE ON {DEVICE}")
print("   (Active Laws: Trauma + Mortality + Ischemia + Blindness)")

# =====================
# THE BODY (Unified Physics)
# =====================
class RiftWorldBody(nn.Module):
    def __init__(self):
        super().__init__()
        # Laplacian kernel for diffusion
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
        # --- LAW 1: MEMORY (Traces) ---
        self.trace_fast = self.trace_fast * 0.8 + cue * 1.0
        self.trace_slow = self.trace_slow * 0.98 + cue * 0.5

        # --- LAW 2: MUSCLE & ISCHEMIA ---
        # Charging is slower if damaged (Scarring)
        efficiency = torch.clamp(self.health, 0.1, 1.0)
        charge_rate = 0.05 * efficiency * prepare_signal

        # Leaks slightly
        self.brace_charge = (self.brace_charge + charge_rate) * 0.98
        self.brace_charge = torch.clamp(self.brace_charge, 0.0, 1.0)

        # ISCHEMIA: Holding tension > 0.3 rots the body
        ischemic_damage = torch.tensor(0.0, device=DEVICE)
        if self.brace_charge > 0.3:
            ischemic_damage = (self.brace_charge - 0.3) * 0.05
            self.health = self.health - ischemic_damage

        # --- LAW 3: SHIELDING & PHYSICS ---
        shield = (1.0 - self.brace_charge) # 1.0 = Exposed, 0.0 = Protected

        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -1.0 * self.u
        # Force is dampened by the shield
        accel = diffusion + restoring + (external_force * shield) - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # --- LAW 4: TRAUMA ---
        deformation = self.u.abs().max()
        pain_signal = torch.tensor(0.0, device=DEVICE)

        if deformation > 0.1:
            trauma = deformation * 0.2
            self.health = self.health - trauma
            pain_signal = trauma * 100.0 # High pain for damage

        # --- LAW 5: MORTALITY ---
        dead = False
        if self.health <= 0.0:
            dead = True
            self.health = torch.tensor(0.0, device=DEVICE)
            pain_signal += 500.0 # Final penalty

        # Cost includes Ischemia (The "Burn")
        pain_signal += ischemic_damage * 50.0

        return pain_signal, self.get_sensor_data(), dead

    def get_sensor_data(self):
        # Raw Data
        raw_state = torch.stack([
            self.u.abs().mean().view(1),
            self.trace_fast.mean().view(1),
            self.brace_charge.view(1),
            self.health.view(1)
        ]).squeeze()

        # --- LAW 6: BLINDNESS (Path F) ---
        # High Tension = High Noise
        # If Charge is 1.0, Noise is 1.0 (Signal is destroyed)
        noise_level = self.brace_charge * 1.0
        noise = torch.randn_like(raw_state) * noise_level * 0.5

        perceived_state = raw_state + noise

        # The organism sees the noisy version, not the truth
        return perceived_state

# =====================
# THE CONTROLLER (Reflex)
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
    body = RiftWorldBody().to(DEVICE)
    spine = SpinalReflex().to(DEVICE)
    optimizer = optim.Adam(spine.parameters(), lr=0.005)

    print("\n⚔️ TRAINING: Survival in the RIFT-WORLD")
    print("   (Paranoia causes Blindness & Ischemia. Laziness causes Death.)")

    for episode in range(STEPS_TRAIN):
        body.reset()
        total_loss = 0

        impact_time = random.randint(40, 80)
        cue_time = impact_time - 15

        tension_history = []
        health_history = []

        for t in range(STEPS_PER_EPISODE):
            # WORLD EVENTS
            cue = torch.zeros_like(body.u)
            if t == cue_time: cue[:,:,16-4:16+4,16-4:16+4] = 1.0

            force = torch.zeros_like(body.u)
            if t >= impact_time and t < impact_time + 5:
                force[:,:,16-2:16+2,16-2:16+2] = 20.0 # Lethal

            # SENSE (Noisy)
            perceived_state = body.get_sensor_data()

            # REACT
            prepare_signal = spine(perceived_state).squeeze()

            # LOG
            tension_history.append(body.brace_charge.item())
            health_history.append(body.health.item())

            # STEP
            pain, _, dead = body.step(force, cue, prepare_signal)
            total_loss += pain

            if dead:
                break

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if episode % 500 == 0:
            if len(tension_history) > impact_time:
                tens = tension_history[impact_time-1]
                hlth = health_history[impact_time]
            else:
                tens = tension_history[-1]
                hlth = 0.0
            print(f"   Episode {episode:04d} | Pain: {total_loss.item():.2f} | Tension@Hit: {tens:.3f} | Health: {hlth:.2f}")

    print("\n🧪 FINAL DIAGNOSTIC: The Blindness Test")

    body.reset()
    impact_t = 60
    cue_t = 45

    tensions = []
    healths = []
    inputs_seen = []

    for t in range(STEPS_PER_EPISODE):
        cue = torch.zeros_like(body.u)
        if t == cue_t: cue[:,:,16-4:16+4,16-4:16+4] = 1.0

        force = torch.zeros_like(body.u)
        if t >= impact_t and t < impact_t + 5: force[:,:,16-2:16+2,16-2:16+2] = 20.0

        state = body.get_sensor_data()
        inputs_seen.append(state[1].item()) # Log the FastTrace (Cue) visibility

        with torch.no_grad():
            action = spine(state).squeeze()

        tensions.append(body.brace_charge.item())
        healths.append(body.health.item())

        _, _, dead = body.step(force, cue, action)
        if dead: break

    # ANALYSIS
    print("\n   Timeline (Survival + Blindness):")
    print("   T | Cue | Hit | Tension | Vision (Cue Signal) | Health")

    limit = min(80, len(tensions))
    for t in range(40, limit):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "

        # Visualize Signal Clarity
        # If Tension is high, Vision should be messy
        vis_val = inputs_seen[t]

        print(f"   {t} | {c} | {h} | {tensions[t]:.3f}   | {vis_val:.4f}            | {healths[t]:.2f}")

    # Success Condition:
    # 1. Tension LOW before cue (to see it)
    # 2. Tension HIGH at impact (to survive)
    low_pre_tension = sum(tensions[40:cue_t]) / (cue_t - 40) < 0.3
    high_impact_tension = tensions[impact_t] > 0.7
    survived = healths[-1] > 0.0

    if survived and low_pre_tension and high_impact_tension:
        print("\n   🏆 GLORIOUS SUCCESS: Anticipation Emerged!")
        print("      System relaxed to see, and braced to survive.")
    elif not survived:
        print("\n   💀 FAILURE: Organism died.")
    elif not low_pre_tension:
        print("\n   🐢 PARTIAL: Survived via Paranoia (Always Tense).")
        print("      (Blindness pressure needs to be higher?)")
    else:
        print("\n   ❌ FAILURE: Other.")

if __name__ == "__main__":
    run()