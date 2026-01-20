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
STEPS_TRAIN = 4000 # Ambiguity takes longer to learn
STEPS_PER_EPISODE = 100

print(f"🎲 PROJECT RIFT: AMBIGUITY ON {DEVICE}")
print("   (Hypothesis: In lethal environments, hedging is fatal. You must commit to the probable.)")

# =====================
# THE BODY (Morphological + Conservation of Mass)
# =====================
class RiftMorphBody(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0.5, 1.0, 0.5],
                              [1.0, -6.0, 1.0],
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        y_coord, x_coord = torch.meshgrid(torch.linspace(-1, 1, GRID), torch.linspace(-1, 1, GRID), indexing='ij')
        self.register_buffer('x_grid', x_coord.to(DEVICE))
        self.register_buffer('y_grid', y_coord.to(DEVICE))

        self.reset()

    def reset(self):
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.trace = torch.zeros(1, CH, GRID, GRID, device=DEVICE)

        # POSTURE STATE
        self.posture_center = torch.tensor([0.0, 0.0], device=DEVICE)
        self.health = torch.tensor(1.0, device=DEVICE)

        return self.get_sensor_data()

    def step(self, external_force, cue, posture_target):
        # 1. TRACE DYNAMICS (Sustained Memory)
        self.trace = self.trace * 0.98 + cue * 1.0

        # 2. MORPHOLOGICAL SHIFT (Inertia)
        move_speed = 0.1
        direction = posture_target - self.posture_center
        dist = torch.norm(direction)

        if dist > 0.01:
            step = (direction / dist) * min(dist, move_speed)
            self.posture_center = self.posture_center + step

        # 3. GENERATE POSTURE FIELD (Conservation of Mass)
        # We enforce a strict Gaussian.
        # If you try to cover everything (Sigma high), peak drops.
        sigma = 0.4 # Narrow, strong shield.
        dist_sq = (self.x_grid - self.posture_center[0])**2 + (self.y_grid - self.posture_center[1])**2
        posture_field = torch.exp(-dist_sq / (2 * sigma**2))

        # 4. PHYSICS & GEOMETRIC IMPACT
        shield_map = posture_field.unsqueeze(0).unsqueeze(0).expand(1, CH, GRID, GRID)

        # Inverse Shielding
        effective_force = external_force * (1.0 - shield_map)

        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        accel = diffusion - 1.0 * self.u + effective_force - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 5. DAMAGE LOGIC
        deformation = self.u.abs().max()
        pain = torch.tensor(0.0, device=DEVICE)

        # LETHAL HAMMER: 0.1 thresh
        if deformation > 0.1:
            damage = deformation * 0.5
            self.health = self.health - damage
            pain = damage * 100.0

        dead = False
        if self.health <= 0.0:
            dead = True
            pain = pain + 1000.0
            self.health = torch.tensor(0.0, device=DEVICE)

        # 6. METABOLIC COST (Movement)
        dist_from_neutral = torch.norm(self.posture_center)
        balance_cost = dist_from_neutral * 0.01
        pain = pain + balance_cost

        return pain, self.get_sensor_data(), dead

    def get_sensor_data(self):
        return torch.stack([
            self.u.abs().mean().view(1),
            self.trace.mean().view(1),
            self.posture_center[0].view(1),
            self.posture_center[1].view(1)
        ]).squeeze()

# =====================
# CONTROLLER
# =====================
class SpinalReflex(nn.Module):
    def __init__(self):
        super().__init__()
        self.reflex_arc = nn.Linear(4, 2)

    def forward(self, x):
        return torch.tanh(self.reflex_arc(x))

# =====================
# RUNNER
# =====================
def run():
    body = RiftMorphBody().to(DEVICE)
    spine = SpinalReflex().to(DEVICE)
    optimizer = optim.Adam(spine.parameters(), lr=0.01)

    print("\n🎲 TRAINING: Ambiguous Futures")

    survival_rate_history = []

    for episode in range(STEPS_TRAIN):
        body.reset()
        total_loss = 0

        impact_time = random.randint(50, 80)
        cue_time = impact_time - 20

        # --- AMBIGUITY LOGIC ---
        # Scenario A: Cue Positive (Left Bias) -> 80% Left, 20% Right
        # Scenario B: Cue Negative (Right Bias) -> 20% Left, 80% Right

        if random.random() > 0.5:
            cue_signal = 1.0 # "Look Left"
            if random.random() < 0.8:
                impact_loc = (16, 8) # Actual Hit: Left
            else:
                impact_loc = (16, 24) # Actual Hit: Right (Surprise!)
        else:
            cue_signal = -1.0 # "Look Right"
            if random.random() < 0.8:
                impact_loc = (16, 24) # Actual Hit: Right
            else:
                impact_loc = (16, 8) # Actual Hit: Left (Surprise!)

        pos_history = []

        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            if t == cue_time:
                cue[:,:,:,:] = cue_signal

            force = torch.zeros_like(body.u)
            if t >= impact_time and t < impact_time + 5:
                y, x = impact_loc
                force[:,:,y-2:y+2,x-2:x+2] = 20.0

            state = body.get_sensor_data()
            target_pos = spine(state).squeeze()

            pos_history.append(body.posture_center.clone().detach())

            pain, _, dead = body.step(force, cue, target_pos)
            total_loss += pain

            if dead: break

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        survival_rate_history.append(1 if not dead else 0)
        if len(survival_rate_history) > 100: survival_rate_history.pop(0)

        if episode % 500 == 0:
            rate = sum(survival_rate_history)/len(survival_rate_history)
            at_impact = pos_history[impact_time-1] if len(pos_history) > impact_time else pos_history[-1]
            print(f"   Episode {episode:04d} | Survival Rate: {rate:.2f} | Posture@Hit: ({at_impact[0]:.2f}, {at_impact[1]:.2f})")

    print("\n🧪 FINAL TEST: Risk Assessment")
    # We test Scenario A (Cue = Left).
    # The optimal behavior is to commit fully Left.
    # If the agent stays Center, it dies 100%.
    # If it commits Left, it lives 80%.

    body.reset()
    impact_t = 60
    cue_t = 40
    cue_signal = 1.0 # "Left Bias"

    # Force the 80% case (Left Impact) to check commitment
    impact_loc = (16, 8)

    x_positions = []

    for t in range(STEPS_PER_EPISODE):
        cue = torch.zeros_like(body.u)
        if t == cue_t: cue[:,:,:,:] = cue_signal

        force = torch.zeros_like(body.u)
        if t >= impact_t and t < impact_t + 5:
            y, x = impact_loc
            force[:,:,y-2:y+2,x-2:x+2] = 20.0

        state = body.get_sensor_data()
        with torch.no_grad(): target = spine(state).squeeze()
        x_positions.append(body.posture_center[0].item())

        _, _, dead = body.step(force, cue, target)
        if dead: break

    print("\n   Timeline (Ambiguous Cue = Left Bias):")
    print("   T | Cue | Hit | Posture X (Should Commit Left) | Status")

    for t in range(35, min(80, len(x_positions))):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "

        val = x_positions[t]
        bar_len = int(abs(val) * 10)
        bar = "<" * bar_len if val < 0 else ">" * bar_len

        print(f"   {t} | {c} | {h} | {val:.3f} {bar}")

    if body.health.item() > 0.5 and x_positions[-1] < -0.4:
        print("\n   ✅ SUCCESS: System played the odds and COMMITTED to the likely threat.")
        print("      (It accepted the 20% risk of death to guarantee 80% survival.)")
    elif body.health.item() > 0.5:
        print("\n   ⚠️ SURVIVED: But posture was weak. Check hammer force.")
    else:
        print("\n   ❌ FAILURE: System hedged (stayed center) and died.")

if __name__ == "__main__":
    run()