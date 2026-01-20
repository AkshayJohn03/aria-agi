import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# CONFIG
# ======================
GRID = 32
CH = 8
DT = 0.05
STEPS = 120
EPISODES = 2000

print("🧬 PROJECT RIFT: MINIMAL SPECIATION ON", DEVICE)
print("   (Hypothesis: Hard physiological exclusion forces divergence.)")

# ======================
# BIOMES
# ======================
def biome(mode):
    if mode == "CRUSHER":
        return {"width": 14, "force": 8.0}
    if mode == "SNIPER":
        return {"width": 2, "force": 60.0}

# ======================
# BODY
# ======================
class RiftBody:
    def __init__(self):
        self.lap = torch.tensor(
            [[[[0.5,1.0,0.5],[1.0,-6.0,1.0],[0.5,1.0,0.5]]]],
            device=DEVICE
        ).repeat(CH,1,1,1)

        y, x = torch.meshgrid(
            torch.linspace(-1,1,GRID),
            torch.linspace(-1,1,GRID),
            indexing="ij"
        )
        self.xg, self.yg = x.to(DEVICE), y.to(DEVICE)

        # Genetics
        self.health_max = 1.0
        self.battery_max = 1.0

        self.reset()

    def reset(self):
        self.u = torch.zeros(1,CH,GRID,GRID,device=DEVICE)
        self.v = torch.zeros_like(self.u)
        self.trace = torch.zeros_like(self.u)

        self.pos = torch.tensor([0.0,0.0], device=DEVICE)
        self.vel = torch.zeros_like(self.pos)

        self.health = self.health_max
        self.battery = self.battery_max * 0.5

        self.used_impulse = False
        self.took_damage = False
        self.died = False

    @torch.no_grad()
    def step(self, force, cue, tilt, fire):
        self.trace = self.trace * 0.9 + cue

        # Recharge
        self.battery = min(self.battery + 0.03, self.battery_max)

        # =========================
        # HARD PHYSIOLOGICAL EXCLUSION
        # =========================

        impulse = 0.0

        # ❌ Tanks cannot dodge
        if self.health_max < 2.0:
            if fire > 0.5 and self.battery > 0.6 * self.battery_max:
                impulse = (1 if tilt > 0 else -1) * self.battery * 6.0
                self.battery = 0.0
                self.used_impulse = True

        # Motion
        self.vel[0] += impulse * DT
        self.pos += self.vel * DT
        self.pos = torch.clamp(self.pos, -1.0, 1.0)

        # =========================
        # PHYSICS FIELD
        # =========================
        dist = (self.xg - self.pos[0])**2 + (self.yg - self.pos[1])**2
        shield = torch.exp(-dist/(2*0.4**2)).unsqueeze(0).unsqueeze(0)

        eff = force * (1.0 - shield)
        diff = F.conv2d(self.u, self.lap, padding=1, groups=CH)

        self.v += (diff - self.u + eff - 0.1*self.v)*DT
        self.u += self.v*DT

        deform = self.u.abs().max().item()

        # ❌ Speedsters are fragile
        if self.health_max < 2.0 and deform > 0.05:
            self.health = 0.0

        # ❌ Tanks survive damage
        if self.health_max >= 2.0 and deform > 0.1:
            self.health -= deform * 0.2
            self.took_damage = True

        if self.health <= 0:
            self.died = True

        return self.died

    def evolve(self):
        # DARWINIAN FILTER
        if self.died:
            self.health_max = max(1.0, self.health_max * 0.95)
            self.battery_max = max(1.0, self.battery_max * 0.95)
            return

        # Survivor adaptation
        if self.took_damage:
            self.health_max += 0.1

        if self.used_impulse:
            self.battery_max += 0.1

    def state(self):
        return torch.tensor([
            self.pos[0].item(),
            self.battery / self.battery_max,
            self.health / self.health_max
        ], device=DEVICE)

# ======================
# BRAIN (MINIMAL)
# ======================
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3,2)

    def forward(self,x):
        o = self.fc(x)
        return torch.tanh(o[0]), torch.sigmoid(o[1])

# ======================
# RUNNER
# ======================
def run_universe(name):
    print(f"\n🌍 UNIVERSE: {name}")
    env = biome(name)

    body = RiftBody()
    brain = Brain().to(DEVICE)
    opt = optim.Adam(brain.parameters(), lr=0.01)

    for ep in range(EPISODES):
        body.reset()

        impact = random.randint(70,100)
        cue_t = impact - 30

        for t in range(STEPS):
            cue = torch.zeros_like(body.u)
            if t == cue_t:
                cue[:] = 1.0

            force = torch.zeros_like(cue)
            if impact <= t < impact+5:
                w = env["width"]//2
                cx = 16
                force[:,:,14:18,cx-w:cx+w] = env["force"]

            tilt, fire = brain(body.state())
            dead = body.step(force, cue, tilt.item(), fire.item())

            loss = (tilt**2 + fire**2) * (1.0 if dead else 0.01)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if dead:
                break

        body.evolve()

        if ep % 500 == 0:
            print(f"EP {ep:04d} | H={body.health_max:.2f} | B={body.battery_max:.2f}")

    print(f"🏁 FINAL [{name}] → H={body.health_max:.2f} | B={body.battery_max:.2f}")

if __name__ == "__main__":
    run_universe("CRUSHER")
    run_universe("SNIPER")
