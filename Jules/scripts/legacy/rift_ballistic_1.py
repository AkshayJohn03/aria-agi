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
STEPS_TRAIN = 1500
STEPS_PER_EPISODE = 100

print(f"🏹 PROJECT RIFT: BALLISTIC-BRACE (LETHAL) ON {DEVICE}")
print("   (Hypothesis: Making the hammer heavier forces the shield to be used.)")

# =====================
# THE BODY
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
        return self.get_sensor_data()

    def step(self, external_force, cue, prepare_signal):
        # 1. UPDATE TRACES
        self.trace_fast = self.trace_fast * 0.8 + cue * 1.0
        self.trace_slow = self.trace_slow * 0.98 + cue * 0.5

        # 2. BALLISTIC MUSCLE
        charge_rate = 0.05 * prepare_signal
        leak_rate = 0.98
        self.brace_charge = (self.brace_charge + charge_rate) * leak_rate
        self.brace_charge = torch.clamp(self.brace_charge, 0.0, 1.0)

        # 3. PHYSICS
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -1.0 * self.u
        accel = diffusion + restoring + external_force - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 4. IMPACT LOGIC
        deformation = self.u.abs().max()
        pain_signal = torch.tensor(0.0, device=DEVICE)

        # LETHAL THRESHOLD: Trigger pain much earlier (0.1 instead of 0.5)
        if deformation > 0.1:
            # CHECK COMMITMENT
            if self.brace_charge > 0.6:
                # SUCCESS: Blocked!
                self.brace_charge = torch.tensor(0.0, device=DEVICE)
                pain_signal = torch.tensor(0.0, device=DEVICE)
            else:
                # FAILURE: DEATH PENALTY
                # We scale pain by deformation to give a gradient to the fear
                pain_signal = deformation * 100.0

        # 5. COST
        metabolic_cost = prepare_signal * 0.1
        total_pain = pain_signal + metabolic_cost

        return total_pain, self.get_sensor_data()

    def get_sensor_data(self):
        return torch.stack([
            self.u.abs().mean().view(1),
            self.trace_fast.mean().view(1),
            self.trace_slow.mean().view(1),
            self.brace_charge.view(1)
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

    print("\n🧠 TUNING: Spinal Reflexes")

    for episode in range(STEPS_TRAIN):
        body.reset()
        total_loss = 0

        impact_time = random.randint(40, 80)
        cue_time = impact_time - 15

        tension_history = []
        action_history = []

        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            if t == cue_time: cue[:,:,16-4:16+4,16-4:16+4] = 1.0

            force = torch.zeros_like(body.u)
            # LETHAL HAMMER: 5 frames of Force 20.0
            if t >= impact_time and t < impact_time + 5:
                force[:,:,16-2:16+2,16-2:16+2] = 20.0

            sensor_state = body.get_sensor_data()
            prepare_signal = spine(sensor_state).squeeze()

            tension_history.append(body.brace_charge.item())
            action_history.append(prepare_signal.item())

            pain, _ = body.step(force, cue, prepare_signal)
            total_loss += pain

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if episode % 200 == 0:
            tension_at_impact = tension_history[impact_time-1]
            print(f"   Episode {episode:04d} | Pain: {total_loss.item():.2f} | Tension @ Impact: {tension_at_impact:.3f}")

    print("\n🧪 FINAL TEST: The Reflex Arc")

    body.reset()
    impact_t = 60
    cue_t = 45

    tensions = []
    actions = []

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
        actions.append(action.item())

        body.step(force, cue, action)

    print("\n   Timeline (T=40 to T=75):")
    print("   T | Cue | Hit | Action (Charge) | Muscle Tension")
    for t in range(40, 76):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "
        bar = "#" * int(tensions[t] * 10)
        print(f"   {t} | {c} | {h} | {actions[t]:.3f}           | {tensions[t]:.3f} {bar}")

    if tensions[impact_t] > 0.6:
        print("\n   ✅ SUCCESS: Reflex arc established. Body charges proactively.")
    else:
        print("\n   ❌ FAILURE: No anticipation.")

if __name__ == "__main__":
    run()