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
STEPS = 120
EPISODES = 3000

print("🌍 PROJECT RIFT: ECOLOGY ON", DEVICE)
print("   (Hypothesis: Different environments induce different stable morphologies.)")

# ======================================================
# BIOMES
# ======================================================
BIOMES = {
    "CRUSHER": {
        "hammer_width": 10,
        "hammer_force": 8.0,
        "desc": "Wide, unavoidable impact (Tank favored)"
    },
    "SNIPER": {
        "hammer_width": 3,
        "hammer_force": 30.0,
        "desc": "Narrow, lethal impact (Speed favored)"
    },
    "CHAOS": {
        "hammer_width": 4,
        "hammer_force": 15.0,
        "desc": "Repeated cue reversals (Stability favored)"
    }
}

# ======================================================
# BODY (NON-DIFFERENTIABLE, EVOLVING)
# ======================================================
class RiftEcologyBody:
    def __init__(self):
        lap = torch.tensor(
            [[[[0.5,1.0,0.5],[1.0,-6.0,1.0],[0.5,1.0,0.5]]]],
            device=DEVICE
        ).repeat(CH,1,1,1)
        self.lap = lap

        y, x = torch.meshgrid(
            torch.linspace(-1,1,GRID),
            torch.linspace(-1,1,GRID),
            indexing="ij"
        )
        self.xg = x.to(DEVICE)
        self.yg = y.to(DEVICE)

        # Evolvable traits
        self.health_max = 1.0
        self.battery_max = 1.0
        self.barrier = 2.0

        self.reset()

    def reset(self):
        self.u = torch.zeros(1,CH,GRID,GRID,device=DEVICE)
        self.v = torch.zeros_like(self.u)
        self.trace = torch.zeros_like(self.u)

        self.pos = torch.tensor([0.0,0.0],device=DEVICE)
        self.vel = torch.zeros_like(self.pos)

        self.health = self.health_max
        self.battery = self.battery_max * 0.5
        self.stability = 1.0

        self.used_impulse = False
        self.stressed = False

    @torch.no_grad()
    def step(self, force, cue, melt, tilt, fire):
        # Trace
        self.trace = self.trace * 0.85 + cue
        belief = self.trace.mean().item()
        reality = self.pos[0].item()

        # Dissonance
        if abs(belief) > 0.1 and belief * reality < 0:
            self.stability -= 0.04
            self.stressed = True

        self.stability = max(0.0, min(1.0, self.stability))

        # Battery
        self.battery = min(self.battery + 0.02, self.battery_max)

        impulse = 0.0
        if fire > 0.5 and self.battery > 0.6 * self.battery_max:
            impulse = (1 if tilt > 0 else -1) * self.battery * 6.0
            self.battery = 0.0
            self.used_impulse = True

        # Forces
        px = self.pos[0].item()
        fx = -4 * self.barrier * px * (px**2 - 0.6**2) + tilt*0.4 + impulse
        fy = -2 * self.pos[1].item()

        damp = 0.5 + (1.0 - self.stability)*0.5

        self.vel += torch.tensor([fx,fy],device=DEVICE)*DT
        self.vel *= (1 - damp*DT)
        self.pos = torch.clamp(self.pos + self.vel*DT, -1.0,1.0)

        # Shield
        dist = (self.xg-self.pos[0])**2 + (self.yg-self.pos[1])**2
        shield = torch.exp(-dist/(2*0.4**2))
        shield = shield.unsqueeze(0).unsqueeze(0).expand(1,CH,GRID,GRID)

        eff = force * (1.0 - shield)
        diff = F.conv2d(self.u,self.lap,padding=1,groups=CH)
        self.v += (diff - self.u + eff - 0.1*self.v)*DT
        self.u += self.v*DT

        dmg = 0.0
        deform = self.u.abs().max().item()
        if deform > 0.1:
            dmg = deform * 0.4
            self.health -= dmg

        return self.health <= 0, dmg

    def evolve(self):
        if self.used_impulse:
            self.battery_max += 0.05
        if self.stressed and self.health > 0:
            self.health_max += 0.05
        if self.health <= 0:
            self.barrier = max(1.0, self.barrier * 0.95)

    def state(self):
        return torch.tensor([
            self.trace.mean().item(),
            self.pos[0].item(),
            self.battery/self.battery_max,
            self.stability,
            self.health/self.health_max,
            self.battery_max
        ],device=DEVICE)

# ======================================================
# BRAIN (SIMPLE, MEMORYLESS)
# ======================================================
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6,3)

    def forward(self,x):
        o = self.fc(x)
        return torch.sigmoid(o[0]), torch.tanh(o[1]), torch.sigmoid(o[2])

# ======================================================
# RUNNER
# ======================================================
def run():
    body = RiftEcologyBody()
    brain = Brain().to(DEVICE)
    opt = optim.Adam(brain.parameters(),lr=0.01)

    biome_names = list(BIOMES.keys())

    for ep in range(EPISODES):
        body.reset()
        biome = BIOMES[random.choice(biome_names)]

        impact_t = random.randint(80,100)
        cue1, cue2 = impact_t-60, impact_t-20

        for t in range(STEPS):
            cue = torch.zeros_like(body.u)
            if t == cue1: cue[:] = 1.0
            if t == cue2: cue[:] = -2.0

            force = torch.zeros_like(cue)
            if impact_t <= t < impact_t+5:
                w = biome["hammer_width"]
                force[:,:,16-w:16+w,16-w:16+w] = biome["hammer_force"]

            state = body.state()
            melt, tilt, fire = brain(state)

            dead, dmg = body.step(
                force, cue,
                melt.item(), tilt.item(), fire.item()
            )

            pain = (melt**2 + tilt**2 + fire**2) * dmg
            pain = pain + melt*0.05 + torch.abs(tilt)*0.01 + fire*0.01

            opt.zero_grad()
            pain.backward()
            opt.step()

            if dead:
                break

        body.evolve()

        if ep % 200 == 0:
            print(
                f"EP {ep:04d} | "
                f"Biome {biome['desc'][:12]} | "
                f"H_max {body.health_max:.2f} | "
                f"B_max {body.battery_max:.2f}"
            )

if __name__ == "__main__":
    run()
