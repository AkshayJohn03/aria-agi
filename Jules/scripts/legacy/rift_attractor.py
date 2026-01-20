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

print(f"🕳️ PROJECT RIFT: ATTRACTOR (FIXED) ON {DEVICE}")
print("   (Hypothesis: Decisions persist because gravity holds them in stable basins.)")

# =====================
# THE BODY (Attractor Landscape)
# =====================
class RiftAttractorBody(nn.Module):
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
        self.posture_vel = torch.tensor([0.0, 0.0], device=DEVICE) # Inertia
        self.health = torch.tensor(1.0, device=DEVICE)

        return self.get_sensor_data()

    def step(self, external_force, cue, landscape_tilt):
        # 1. TRACE DYNAMICS
        self.trace = self.trace * 0.9 + cue * 1.0

        # 2. ATTRACTOR DYNAMICS
        # Double Well Potential V(x) = k * (x^2 - a^2)^2
        target_well = 0.6
        stiffness = 2.0

        px = self.posture_center[0]
        py = self.posture_center[1]

        # Force = -dV/dx
        force_well_x = -4.0 * stiffness * px * (px**2 - target_well**2)
        force_well_y = -2.0 * py

        # 3. BRAIN TILT (Linear slope addition)
        tilt_strength = 2.0
        force_tilt = landscape_tilt * tilt_strength

        # 4. INTEGRATION (Strict Out-of-Place Updates)
        total_force_x = force_well_x + force_tilt[0]
        total_force_y = force_well_y + force_tilt[1]

        damping = 0.2

        # Update Velocity (Out-of-place)
        new_vel_x = self.posture_vel[0] + (total_force_x - self.posture_vel[0] * damping) * DT
        new_vel_y = self.posture_vel[1] + (total_force_y - self.posture_vel[1] * damping) * DT
        self.posture_vel = torch.stack([new_vel_x, new_vel_y])

        # Update Position (Out-of-place)
        self.posture_center = self.posture_center + self.posture_vel * DT

        # Clamp
        self.posture_center = torch.clamp(self.posture_center, -1.0, 1.0)

        # 5. POSTURE FIELD GENERATION
        sigma = 0.4
        dist_sq = (self.x_grid - self.posture_center[0])**2 + (self.y_grid - self.posture_center[1])**2
        posture_field = torch.exp(-dist_sq / (2 * sigma**2))
        shield_map = posture_field.unsqueeze(0).unsqueeze(0).expand(1, CH, GRID, GRID)

        # 6. PHYSICS SIMULATION
        effective_force = external_force * (1.0 - shield_map)
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)

        # Physics Step (Out-of-place)
        accel = diffusion - 1.0 * self.u + effective_force - 0.1 * self.v
        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 7. DAMAGE LOGIC
        deformation = self.u.abs().max()
        pain = torch.tensor(0.0, device=DEVICE)

        if deformation > 0.1:
            damage = deformation * 0.5
            self.health = self.health - damage
            pain = damage * 100.0

        dead = False
        if self.health <= 0.0:
            dead = True
            pain = pain + 1000.0
            self.health = torch.tensor(0.0, device=DEVICE)

        # 8. METABOLIC COST
        # Cost only for TILTING the landscape, not for sitting in the well.
        tilt_cost = torch.norm(landscape_tilt) * 0.5
        pain = pain + tilt_cost

        return pain, self.get_sensor_data(), dead

    def get_sensor_data(self):
        # Force views to prevent scalar/vector mismatch
        return torch.stack([
            self.u.abs().mean().view(1),
            self.trace.mean().view(1),
            self.posture_center[0].view(1),
            self.posture_center[1].view(1)
        ]).squeeze()

# =====================
# THE CONTROLLER
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
    body = RiftAttractorBody().to(DEVICE)
    spine = SpinalReflex().to(DEVICE)
    optimizer = optim.Adam(spine.parameters(), lr=0.01)

    print("\n🕳️ TRAINING: Attractor Basins")

    for episode in range(STEPS_TRAIN):
        body.reset()
        total_loss = 0

        impact_time = random.randint(50, 80)
        cue_time = impact_time - 25

        cue_signal = 1.0
        impact_loc = (16, 8)

        pos_history = []
        tilt_history = []

        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            if t == cue_time: cue[:,:,:,:] = cue_signal

            force = torch.zeros_like(body.u)
            if t >= impact_time and t < impact_time + 5:
                y, x = impact_loc
                force[:,:,y-2:y+2,x-2:x+2] = 20.0

            state = body.get_sensor_data()
            tilt = spine(state).squeeze()

            pos_history.append(body.posture_center.clone().detach())
            tilt_history.append(torch.norm(tilt).item())

            pain, _, dead = body.step(force, cue, tilt)
            total_loss += pain

            if dead: break

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if episode % 500 == 0:
            at_hit = pos_history[impact_time-1] if len(pos_history) > impact_time else pos_history[-1]
            print(f"   Episode {episode:04d} | Pain: {total_loss.item():.2f} | Posture@Hit: ({at_hit[0]:.2f}, {at_hit[1]:.2f}) | Health: {body.health.item():.2f}")

    print("\n🧪 FINAL TEST: The Zero-Cost Hold")
    body.reset()

    impact_t = 80
    cue_t = 20
    cue_signal = 1.0
    impact_loc = (16, 8)

    x_positions = []
    energy_burn = []

    for t in range(STEPS_PER_EPISODE):
        cue = torch.zeros_like(body.u)
        if t == cue_t: cue[:,:,:,:] = cue_signal

        force = torch.zeros_like(body.u)
        if t >= impact_t and t < impact_t + 5:
            y, x = impact_loc
            force[:,:,y-2:y+2,x-2:x+2] = 20.0

        state = body.get_sensor_data()
        with torch.no_grad(): tilt = spine(state).squeeze()

        x_positions.append(body.posture_center[0].item())
        energy_burn.append(torch.norm(tilt).item())

        _, _, dead = body.step(force, cue, tilt)
        if dead: break

    print("\n   Timeline (Attractor Dynamics):")
    print("   T | Cue | Hit | Posture X (Basin=-0.6) | Brain Effort (Tilt) | Status")

    for t in range(15, min(90, len(x_positions))):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "

        pos = x_positions[t]
        effort = energy_burn[t]

        bar_pos = "<" * int(abs(pos)*10)
        bar_eff = "!" * int(effort*10)

        print(f"   {t} | {c} | {h} | {pos:.3f} {bar_pos:10} | {effort:.3f} {bar_eff}")

    hold_effort = sum(energy_burn[40:70]) / 30.0
    final_pos = x_positions[impact_t-1]

    if final_pos < -0.5 and hold_effort < 0.1:
        print("\n   🏆 GLORIOUS SUCCESS: Decision locked in Attractor. Brain relaxed. Zero Cost.")
    elif final_pos < -0.5:
        print("\n   ⚠️ PARTIAL: Held posture, but Brain kept pushing (High Cost).")
    else:
        print("\n   ❌ FAILURE: Did not commit to well.")

if __name__ == "__main__":
    run()