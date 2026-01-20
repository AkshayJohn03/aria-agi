import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRID = 32
CH = 8
DT = 0.05

STEPS_TRAIN = 3000
STEPS_PER_EPISODE = 120

print("🧱 PROJECT RIFT: BURDEN ON", DEVICE)
print("   (Hypothesis: Every gain imposes a permanent load.)")

# =====================
# BODY
# =====================
class RiftBurdenBody:
    def __init__(self):
        self.laplacian = torch.tensor(
            [[[[0.5, 1.0, 0.5],
               [1.0, -6.0, 1.0],
               [0.5, 1.0, 0.5]]]],
            device=DEVICE
        ).repeat(CH, 1, 1, 1)

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, GRID),
            torch.linspace(-1, 1, GRID),
            indexing="ij"
        )
        self.x_grid = x.to(DEVICE)
        self.y_grid = y.to(DEVICE)

        # Evolutionary parameters
        self.health_max = 1.0
        self.battery_max = 1.0
        self.barrier_strength = 2.0

        self.reset()

    def reset(self):
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros_like(self.u)
        self.trace = torch.zeros_like(self.u)

        self.posture = torch.tensor([0.0, 0.0], device=DEVICE)
        self.vel = torch.zeros_like(self.posture)

        self.health = self.health_max
        self.stability = 1.0
        self.battery = self.battery_max * 0.5

        self.used_impulse = False
        self.survived_stress = False

    @torch.no_grad()
    def physics_step(self, force, cue, melt, tilt, fire):
        self.trace = self.trace * 0.85 + cue

        belief = self.trace.mean().item()
        reality = self.posture[0].item()

        dissonance = 0.0
        if abs(belief) > 0.1 and belief * reality < 0:
            dissonance = 0.6
            self.stability -= dissonance * 0.04
            self.survived_stress = True

        self.stability = max(0.0, min(1.0, self.stability))

        # Battery charge
        self.battery = min(self.battery + 0.02, self.battery_max)

        # -------- BURDEN --------
        load = 0.02 * self.health_max + 0.05 * self.battery_max

        impulse = 0.0
        if fire > 0.5 and self.battery > 0.6 * self.battery_max:
            direction = 1.0 if tilt > 0 else -1.0
            impulse = direction * self.battery * max(0.5, 6.0 - load)
            self.battery = 0.0
            self.used_impulse = True

        barrier = self.barrier_strength * (1.0 - melt)
        px = self.posture[0].item()
        target = 0.6

        fx = -4 * barrier * px * (px**2 - target**2) + tilt * 0.4 + impulse
        fy = -2.0 * self.posture[1].item()

        damping = 0.5 + load + (1.0 - self.stability) * 0.5

        self.vel += torch.tensor([fx, fy], device=DEVICE) * DT
        self.vel *= (1.0 - damping * DT)
        self.posture = torch.clamp(self.posture + self.vel * DT, -1, 1)

        dist = (self.x_grid - self.posture[0])**2 + (self.y_grid - self.posture[1])**2
        shield = torch.exp(-dist / (2 * 0.4**2))
        shield = shield.unsqueeze(0).unsqueeze(0).expand(1, CH, GRID, GRID)

        eff_force = force * (1.0 - shield)
        diff = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)

        accel = diff - self.u + eff_force - 0.1 * self.v
        self.v += accel * DT
        self.u += self.v * DT

        damage = 0.0
        if self.u.abs().max().item() > 0.1:
            damage = self.u.abs().max().item() * 0.4
            self.health -= damage

        dead = self.health <= 0.0
        return dissonance, damage, dead

    def evolve(self):
        if self.used_impulse:
            self.battery_max += 0.05
        if self.survived_stress and self.health > 0:
            self.health_max += 0.05
        if self.health <= 0:
            self.barrier_strength *= 0.97

    def state(self):
        return torch.tensor([
            self.trace.mean().item(),
            self.posture[0].item(),
            self.battery / self.battery_max,
            self.stability,
            self.health / self.health_max,
            self.battery_max
        ], device=DEVICE)

# =====================
# BRAIN
# =====================
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 3)

    def forward(self, x):
        o = self.fc(x)
        return torch.sigmoid(o[0]), torch.tanh(o[1]), torch.sigmoid(o[2])

# =====================
# RUNNER
# =====================
def run():
    body = RiftBurdenBody()
    brain = Brain().to(DEVICE)
    opt = optim.Adam(brain.parameters(), lr=0.01)

    for ep in range(STEPS_TRAIN):
        body.reset()
        loss = torch.tensor(0.0, device=DEVICE)

        impact = random.randint(80, 110)
        cue1 = impact - 60
        cue2 = impact - 20

        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
            if t == cue1: cue[:] = 1.0
            if t == cue2: cue[:] = -2.0

            force = torch.zeros_like(cue)
            if impact <= t < impact + 5:
                force[:, :, 14:18, 22:26] = 20.0

            state = body.state()
            melt, tilt, fire = brain(state)

            diss, dmg, dead = body.physics_step(
                force, cue, melt.item(), tilt.item(), fire.item()
            )

            pain = (
                (melt**2 + tilt**2 + fire**2) * (diss * 5 + dmg * 100)
                + melt * 0.05
            )

            loss += pain
            if dead:
                break

        opt.zero_grad()
        loss.backward()
        opt.step()
        body.evolve()

        if ep % 200 == 0:
            print(
                f"EP {ep:04d} | "
                f"H_max {body.health_max:.2f} | "
                f"B_max {body.battery_max:.2f}"
            )

if __name__ == "__main__":
    run()
