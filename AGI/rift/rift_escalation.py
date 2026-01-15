import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# CONFIG
# =====================
GRID = 32
CH = 8
DT = 0.05

STEPS_TRAIN = 4000
STEPS_PER_EPISODE = 120

CRITICAL_STABILITY = 0.30
ESCALATION_RATE = 0.04

print(f"🧠🔥 PROJECT RIFT: ESCALATION (FIXED) ON {DEVICE}")
print("   (Hypothesis: Unresolved stress becomes injury.)")

# =====================
# BODY
# =====================
class RiftEscalationBody(nn.Module):
    def __init__(self):
        super().__init__()

        lap = torch.tensor(
            [[[[0.5, 1.0, 0.5],
               [1.0, -6.0, 1.0],
               [0.5, 1.0, 0.5]]]],
            device=DEVICE
        )
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, GRID),
            torch.linspace(-1, 1, GRID),
            indexing="ij"
        )
        self.register_buffer("x_grid", x.to(DEVICE))
        self.register_buffer("y_grid", y.to(DEVICE))

        self.reset()

    def reset(self):
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros_like(self.u)
        self.trace = torch.zeros_like(self.u)

        self.posture = torch.tensor([0.0, 0.0], device=DEVICE)
        self.vel = torch.zeros_like(self.posture)

        self.health = torch.tensor(1.0, device=DEVICE)
        self.stability = torch.tensor(1.0, device=DEVICE)
        self.battery = torch.tensor(0.5, device=DEVICE)

        return self.get_state()

    # =====================
    def step(self, force, cue, melt, tilt, fire):
        # ---- TRACE
        self.trace = self.trace * 0.85 + cue

        belief = self.trace.mean()
        reality = self.posture[0]

        # ---- DISSONANCE
        dissonance = torch.tensor(0.0, device=DEVICE)
        if torch.abs(belief) > 0.1:
            if torch.sign(belief) != torch.sign(reality):
                dissonance = torch.tensor(0.6, device=DEVICE)
                self.stability = self.stability - dissonance * 0.04

        self.stability = torch.clamp(self.stability, 0.0, 1.0)

        # ---- ESCALATION
        escalation_damage = torch.tensor(0.0, device=DEVICE)
        if self.stability < CRITICAL_STABILITY:
            escalation_damage = (CRITICAL_STABILITY - self.stability) * ESCALATION_RATE
            self.health = self.health - escalation_damage

        # ---- BATTERY
        self.battery = torch.clamp(self.battery + 0.02, 0.0, 1.0)

        impulse = torch.tensor([0.0, 0.0], device=DEVICE)
        did_fire = 0.0
        if fire > 0.5 and self.battery > 0.6:
            did_fire = 1.0
            impulse[0] = torch.sign(tilt) * self.battery * 6.0
            self.battery = torch.tensor(0.0, device=DEVICE)

        # ---- LANDSCAPE
        barrier = 2.0 * (1.0 - melt)
        px = self.posture[0]
        target = 0.6

        fx = -4.0 * barrier * px * (px**2 - target**2) + tilt * 0.4 + impulse[0]
        fy = -2.0 * self.posture[1]

        damping = 0.5 + (1.0 - self.stability) * 0.5

        self.vel = self.vel + torch.stack([fx, fy]) * DT - self.vel * damping * DT
        self.posture = torch.clamp(self.posture + self.vel * DT, -1.0, 1.0)

        # ---- FIELD
        dist = (self.x_grid - self.posture[0])**2 + (self.y_grid - self.posture[1])**2
        shield = torch.exp(-dist / (2 * 0.4**2))
        shield = shield.unsqueeze(0).unsqueeze(0).expand(1, CH, GRID, GRID)

        # ---- PHYSICS
        eff_force = force * (1.0 - shield)
        diff = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)

        accel = diff - self.u + eff_force - 0.1 * self.v
        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # ---- DAMAGE
        deform = self.u.abs().max()
        impact_damage = torch.tensor(0.0, device=DEVICE)
        if deform > 0.1:
            impact_damage = deform * 0.4
            self.health = self.health - impact_damage

        self.health = torch.clamp(self.health, 0.0, 1.0)
        dead = self.health <= 0.0

        # ---- PAIN (always differentiable)
        metabolic = melt * 0.05 + torch.abs(tilt) * 0.01 + fire * 0.01
        pain = (
            dissonance * 5.0 +
            escalation_damage * 50.0 +
            impact_damage * 100.0 +
            metabolic
        )

        # ---- SAFE DETACH (OUT-OF-PLACE)
        self.u = self.u.detach()
        self.v = self.v.detach()
        self.trace = self.trace.detach()
        self.posture = self.posture.detach()
        self.vel = self.vel.detach()
        self.health = self.health.detach()
        self.stability = self.stability.detach()
        self.battery = self.battery.detach()

        return pain, self.get_state(), dead, did_fire, dissonance

    def get_state(self):
        return torch.stack([
            self.u.abs().mean(),
            self.trace.mean(),
            self.posture[0],
            self.battery,
            self.stability,
            self.health
        ])

# =====================
# BRAIN
# =====================
class SpinalReflex(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 3)

    def forward(self, x):
        o = self.fc(x)
        return torch.sigmoid(o[0]), torch.tanh(o[1]), torch.sigmoid(o[2])

# =====================
# RUN
# =====================
def run():
    body = RiftEscalationBody().to(DEVICE)
    brain = SpinalReflex().to(DEVICE)
    opt = optim.Adam(brain.parameters(), lr=0.01)

    print("\n🔥 TRAINING: Escalation")

    for ep in range(STEPS_TRAIN):
        body.reset()
        loss = 0.0

        impact_t = random.randint(80, 110)
        cue1 = impact_t - 60
        cue2 = impact_t - 20

        left_first = random.random() > 0.5
        c1, c2 = (1.0, -1.0) if left_first else (-1.0, 1.0)
        impact_x = 24 if left_first else 8

        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            if t == cue1: cue[:] = c1
            if t == cue2: cue[:] = c2 * 2.0

            force = torch.zeros_like(body.u)
            if impact_t <= t < impact_t + 5:
                force[:, :, 16-2:16+2, impact_x-2:impact_x+2] = 20.0

            state = body.get_state()
            melt, tilt, fire = brain(state)

            pain, _, dead, _, _ = body.step(force, cue, melt, tilt, fire)
            loss += pain

            if dead:
                break

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 500 == 0:
            print(
                f"Episode {ep:04d} | "
                f"Stability {body.stability.item():.2f} | "
                f"Health {body.health.item():.2f}"
            )

if __name__ == "__main__":
    run()
