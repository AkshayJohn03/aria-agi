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
STEPS_TRAIN = 2000
STEPS_PER_EPISODE = 100

print(f"💀 PROJECT RIFT: SCAR (MORTAL BODY) ON {DEVICE}")
print("   (Hypothesis: Continuous gradients + Mortality pressure forces agency.)")

# =====================
# THE BODY (Mortal)
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

        # SCALAR STATE
        self.brace_charge = torch.tensor(0.0, device=DEVICE)
        self.health = torch.tensor(1.0, device=DEVICE) # Starts full

        return self.get_sensor_data()

    def step(self, external_force, cue, prepare_signal):
        # 1. TRACES
        self.trace_fast = self.trace_fast * 0.8 + cue * 1.0
        self.trace_slow = self.trace_slow * 0.98 + cue * 0.5

        # 2. MUSCLE (Continuous Charge)
        # Damage (Scarring) reduces your ABILITY to charge.
        # If health is 0.5, you can only charge at 50% speed.
        effective_rate = 0.05 * self.health

        charge_rate = effective_rate * prepare_signal
        leak_rate = 0.98

        self.brace_charge = (self.brace_charge + charge_rate) * leak_rate
        self.brace_charge = torch.clamp(self.brace_charge, 0.0, 1.0)

        # 3. CONTINUOUS SHIELD (The Gradient Fix)
        # Instead of binary If/Else, the shield dampens force linearly.
        # Charge 1.0 = Force 0.0 (Full Block)
        # Charge 0.0 = Force 1.0 (Full Impact)
        shield_factor = (1.0 - self.brace_charge)
        dampened_force = external_force * shield_factor

        # 4. PHYSICS
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -1.0 * self.u
        accel = diffusion + restoring + dampened_force - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 5. DAMAGE & SCARRING
        deformation = self.u.abs().max()
        pain_signal = torch.tensor(0.0, device=DEVICE)

        # Damage logic
        if deformation > 0.1:
            # Damage is proportional to impact
            damage = deformation * 0.1 # Permanent Health loss
            self.health = self.health - damage

            # Pain is high if health drops
            pain_signal = damage * 1000.0

        # Check Mortality
        dead = False
        if self.health <= 0.0:
            dead = True
            pain_signal += 1000.0 # Final death scream
            self.health = torch.tensor(0.0, device=DEVICE)

        # 6. COST
        # Cost is higher if you are damaged (Everything is harder when hurt)
        effort_cost = (1.0 + (1.0 - self.health)) * 0.1
        metabolic_cost = prepare_signal * effort_cost

        total_pain = pain_signal + metabolic_cost

        return total_pain, self.get_sensor_data(), dead

    def get_sensor_data(self):
        # Brain feels its own Health and Charge
        return torch.stack([
            self.u.abs().mean().view(1),
            self.trace_fast.mean().view(1),
            self.brace_charge.view(1),
            self.health.view(1) # Proprioception of mortality
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

    print("\n🧠 TUNING: Survival Instinct")

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
            # LETHAL HAMMER
            if t >= impact_time and t < impact_time + 5:
                force[:,:,16-2:16+2,16-2:16+2] = 20.0

            sensor_state = body.get_sensor_data()
            prepare_signal = spine(sensor_state).squeeze()

            tension_history.append(body.brace_charge.item())
            health_history.append(body.health.item())

            pain, _, dead = body.step(force, cue, prepare_signal)
            total_loss += pain

            # MORTALITY: If dead, loop breaks.
            # You lose the chance to experience the rest of time.
            if dead:
                break

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if episode % 200 == 0:
            # Check status at impact time
            if len(tension_history) > impact_time:
                tens = tension_history[impact_time-1]
                hlth = health_history[impact_time]
            else:
                tens = tension_history[-1] # Dead before impact
                hlth = 0.0

            print(f"   Episode {episode:04d} | Pain: {total_loss.item():.2f} | Tension: {tens:.3f} | Health: {hlth:.2f}")

    print("\n🧪 FINAL TEST: Mortality Check")

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

    print("\n   Timeline (Survival):")
    print("   T | Cue | Hit | Charge | Health")
    limit = min(75, len(tensions))
    for t in range(40, limit):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "
        bar = "#" * int(tensions[t] * 10)
        print(f"   {t} | {c} | {h} | {tensions[t]:.3f}  | {healths[t]:.2f} {bar}")

    if len(healths) > impact_t + 5 and healths[-1] > 0.5:
        print("\n   ✅ SUCCESS: Organism survived impact through anticipation.")
    else:
        print("\n   ❌ FAILURE: Organism died.")

if __name__ == "__main__":
    run()