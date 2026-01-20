import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRID = 32
CH = 8
DT = 0.05
STEPS = 120
EPISODES = 2000 # Shorter, but focused

print("🧬 PROJECT RIFT: SPECIATION ON", DEVICE)
print("   (Hypothesis: Isolation forces distinct morphological strategies.)")

# =====================
# CONFIGURABLE BIOME
# =====================
def get_biome_settings(mode):
    if mode == "CRUSHER":
        return {"width": 12, "force": 8.0, "desc": "TANK WORLD"}
    elif mode == "SNIPER":
        # Note: Force is 50.0. Even high health will struggle.
        # Width is 2. Dodge is easy if you are fast.
        return {"width": 2, "force": 50.0, "desc": "SPEED WORLD"}
    return None

# =====================
# BODY
# =====================
class RiftBody:
    def __init__(self):
        self.lap = torch.tensor([[[[0.5,1.0,0.5],[1.0,-6.0,1.0],[0.5,1.0,0.5]]]], device=DEVICE).repeat(CH,1,1,1)
        y, x = torch.meshgrid(torch.linspace(-1,1,GRID), torch.linspace(-1,1,GRID), indexing="ij")
        self.xg, self.yg = x.to(DEVICE), y.to(DEVICE)

        self.health_max = 1.0
        self.battery_max = 1.0
        self.barrier_base = 2.0
        self.reset()

    def reset(self):
        self.u = torch.zeros(1,CH,GRID,GRID,device=DEVICE)
        self.v = torch.zeros_like(self.u)
        self.trace = torch.zeros_like(self.u)
        self.pos = torch.tensor([0.0,0.0], device=DEVICE)
        self.vel = torch.zeros_like(self.pos)
        self.health = self.health_max
        self.battery = self.battery_max * 0.5
        self.stability = 1.0
        self.used_impulse = False
        self.survived_stress = False

    @torch.no_grad()
    def step(self, force, cue, melt, tilt, fire):
        self.trace = self.trace * 0.85 + cue
        belief = self.trace.mean().item()

        if abs(belief) > 0.1 and belief * self.pos[0].item() < 0:
            self.stability = max(0.0, self.stability - 0.04)
            self.survived_stress = True
        else:
            self.stability = min(1.0, self.stability + 0.01)

        self.battery = min(self.battery + 0.02, self.battery_max)

        # TRADEOFFS
        # 1. Mass Debt: Health makes you heavy.
        mass = 1.0 + 0.8 * (self.health_max - 1.0)
        # 2. Damping Debt: Battery makes you sluggish.
        damping = 0.5 + 0.5 * (self.battery_max - 1.0) + (1.0-self.stability)

        impulse = 0.0
        if fire > 0.5 and self.battery > 0.6 * self.battery_max:
            # Kick strength must be high enough to move the mass
            # But we punish heavy tanks trying to jump
            kick = (8.0 * self.battery) / math.sqrt(mass)
            impulse = (1 if tilt > 0 else -1) * kick
            self.battery = 0.0
            self.used_impulse = True

        barrier = self.barrier_base * (1.0 - melt)
        px = self.pos[0].item()
        f_well = -4 * barrier * px * (px**2 - 0.6**2)
        f_tilt = tilt * 0.4

        accel = (f_well + f_tilt + impulse) / mass

        self.vel[0] += accel * DT
        self.vel *= (1.0 - damping * DT)
        self.pos = torch.clamp(self.pos + self.vel*DT, -1.0, 1.0)

        # Physics Field
        dist = (self.xg - self.pos[0])**2 + (self.yg - self.pos[1])**2
        shield = torch.exp(-dist/(2*0.4**2)).unsqueeze(0).unsqueeze(0).expand(1,CH,GRID,GRID)

        eff = force * (1.0 - shield)
        diff = F.conv2d(self.u, self.lap, padding=1, groups=CH)
        self.v += (diff - self.u + eff - 0.1*self.v)*DT
        self.u += self.v*DT

        dmg = 0.0
        deform = self.u.abs().max().item()
        if deform > 0.1:
            dmg = deform * 0.4
            self.health -= dmg

        return self.health <= 0, dmg

    def evolve(self):
        # Specificity: Only grow what you use
        if self.used_impulse:
            self.battery_max += 0.1
        elif self.battery_max > 1.0: # Atrophy if unused
            self.battery_max -= 0.01

        if self.survived_stress and self.health > 0:
            self.health_max += 0.1
        elif self.health_max > 1.0 and self.health > 0.9 * self.health_max: # Atrophy if too safe
            self.health_max -= 0.01

    def state(self):
        return torch.tensor([
            self.trace.mean().item(), self.pos[0].item(),
            self.battery/self.battery_max, self.stability,
            self.health/self.health_max, self.battery_max
        ], device=DEVICE)

# =====================
# BRAIN
# =====================
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6,3)
    def forward(self,x):
        o = self.fc(x)
        return torch.sigmoid(o[0]), torch.tanh(o[1]), torch.sigmoid(o[2])

import math

def run_universe(mode_name):
    print(f"\n🌌 STARTING UNIVERSE: {mode_name}")
    biome = get_biome_settings(mode_name)

    body = RiftBody()
    brain = Brain().to(DEVICE)
    opt = optim.Adam(brain.parameters(), lr=0.01)

    for ep in range(EPISODES):
        body.reset()

        impact_t = random.randint(80,100)
        cue1, cue2 = impact_t-60, impact_t-20

        if random.random() > 0.5:
            c1, c2 = 1.0, -1.0
            slice_x = (22, 26)
        else:
            c1, c2 = -1.0, 1.0
            slice_x = (6, 10)

        for t in range(STEPS):
            cue = torch.zeros_like(body.u)
            if t == cue1: cue[:] = c1
            if t == cue2: cue[:] = c2 * 2.0

            force = torch.zeros_like(cue)
            if impact_t <= t < impact_t+5:
                w = biome["width"] // 2
                cx = (slice_x[0] + slice_x[1]) // 2
                x1, x2 = max(0, cx-w), min(32, cx+w)
                force[:,:,14:18,x1:x2] = biome["force"]

            state = body.state()
            melt, tilt, fire = brain(state)

            dead, dmg = body.step(force, cue, melt.item(), tilt.item(), fire.item())

            pain = (melt**2 + tilt**2 + fire**2) * (dmg * 10.0) + melt*0.01

            opt.zero_grad()
            pain.backward()
            opt.step()

            if dead: break

        body.evolve()

        if ep % 500 == 0:
            print(f"   EP {ep:04d} | H_max: {body.health_max:.2f} | B_max: {body.battery_max:.2f}")

    print(f"🏁 FINAL SPECIES [{mode_name}]: H={body.health_max:.2f}, B={body.battery_max:.2f}")

if __name__ == "__main__":
    run_universe("CRUSHER")
    run_universe("SNIPER")