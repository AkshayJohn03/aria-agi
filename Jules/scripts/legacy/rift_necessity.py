import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32
CH = 8
DT = 0.05
STEPS = 120
GENERATIONS = 50
POP_SIZE = 15     # Slightly larger pool

print("🔥 PROJECT RIFT: NECESSITY ON", DEVICE)
print("   (Hypothesis: Global stress forces Durability. Local lethal stress forces Agility.)")

# =====================
# BIOMES
# =====================
def get_biome(mode):
    if mode == "CRUSHER":
        return {"type": "GLOBAL", "intensity": 2.0, "desc": "EARTHQUAKE (Tank Required)"}
    elif mode == "SNIPER":
        return {"type": "LOCAL", "intensity": 100.0, "desc": "HEADSHOT (Speed Required)"}
    return None

# =====================
# GENOTYPE
# =====================
class Genotype:
    def __init__(self, h=1.0, b=1.0, barr=2.0):
        self.h_max = h
        self.b_max = b
        self.barrier = barr

    def mutate(self):
        # 20% Mutation rate
        h = self.h_max * random.uniform(0.8, 1.2)
        b = self.b_max * random.uniform(0.8, 1.2)
        return Genotype(h, b, self.barrier)

# =====================
# BODY
# =====================
class RiftBody:
    def __init__(self, genes):
        self.lap = torch.tensor([[[[0.5,1.0,0.5],[1.0,-6.0,1.0],[0.5,1.0,0.5]]]], device=DEVICE).repeat(CH,1,1,1)
        y, x = torch.meshgrid(torch.linspace(-1,1,GRID), torch.linspace(-1,1,GRID), indexing="ij")
        self.xg, self.yg = x.to(DEVICE), y.to(DEVICE)

        self.genes = genes
        self.reset()

    def reset(self):
        self.u = torch.zeros(1,CH,GRID,GRID,device=DEVICE)
        self.v = torch.zeros_like(self.u)
        self.trace = torch.zeros_like(self.u)
        self.pos = torch.tensor([0.0,0.0], device=DEVICE)
        self.vel = torch.zeros_like(self.pos)

        self.health = self.genes.h_max
        self.battery = self.genes.b_max
        self.stability = 1.0

    @torch.no_grad()
    def step(self, force_type, intensity, cue, melt, tilt, fire):
        self.trace = self.trace * 0.85 + cue

        # Stability
        belief = self.trace.mean().item()
        if abs(belief) > 0.1 and belief * self.pos[0].item() < 0:
            self.stability = max(0.0, self.stability - 0.04)

        self.battery = min(self.battery + 0.02, self.genes.b_max)

        # --- BURDEN LAWS ---
        # 1. Mass = Health (Armor is heavy)
        mass = 1.0 + 1.0 * (self.genes.h_max - 1.0) # Steep mass penalty
        # 2. Damping = Battery (Fuel is heavy)
        damping = 0.5 + 0.3 * (self.genes.b_max - 1.0)

        impulse = 0.0
        if fire > 0.5 and self.battery > 0.6 * self.genes.b_max:
            # Force = Battery / sqrt(Mass)
            # Heavy things jump poorly
            kick = (10.0 * self.battery) / math.sqrt(mass)
            impulse = (1 if tilt > 0 else -1) * kick
            self.battery = 0.0

        barrier = self.genes.barrier * (1.0 - melt)
        px = self.pos[0].item()
        f_well = -4 * barrier * px * (px**2 - 0.6**2)
        f_tilt = tilt * 0.4

        accel = (f_well + f_tilt + impulse) / mass

        self.vel[0] += accel * DT
        self.vel *= (1.0 - damping * DT)
        self.pos = torch.clamp(self.pos + self.vel*DT, -1.0, 1.0)

        # DAMAGE LOGIC (The Necessity)
        dmg = 0.0

        if force_type == "GLOBAL":
            # CRUSHER: Damage is unavoidable, but scaled by Health
            # Damage = Intensity / Mass^2 (Square-cube law benefits big things)
            # If you are small (Mass 1.0), you take 2.0 damage/tick -> Death
            # If you are huge (Mass 10.0), you take 0.02 damage/tick -> Survival
            local_pain = intensity / (mass * mass)
            dmg = local_pain * DT

        elif force_type == "LOCAL":
            # SNIPER: Damage is infinite but localized
            # Shield works here
            # Target is at specific location (e.g., Center +/- width)
            # We simulate "Did I get hit?"
            # Let's say Sniper hits center (0.0).
            # If |pos| < 0.2, you die.
            hit_zone = 0.2
            if abs(self.pos[0].item()) < hit_zone: # Didn't dodge
                 dmg = intensity * DT # Massive damage

        self.health -= dmg
        return self.health <= 0, dmg

    def state(self):
        return torch.tensor([
            self.trace.mean().item(), self.pos[0].item(),
            self.battery/self.genes.b_max, self.stability,
            self.health/self.genes.h_max, self.genes.b_max
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

# =====================
# EVOLUTION ENGINE
# =====================
def evaluate_mutant(genes, biome, brain):
    body = RiftBody(genes)

    # Setup Scenario
    # Sniper: Cue at 20, Hit at 80. Must move away from 0.0.
    # Crusher: Just survive.

    cue1 = 20
    survival_time = 0
    total_pain = 0.0

    for t in range(STEPS):
        cue = torch.zeros(1,CH,GRID,GRID,device=DEVICE)
        # Give a cue to move Right (+1)
        if t == cue1: cue[:] = 1.0

        force_active = False
        if t >= 80 and t < 85:
            force_active = True

        state = body.state()
        with torch.no_grad():
            melt, tilt, fire = brain(state)

        f_type = biome["type"] if force_active else "NONE"
        intensity = biome["intensity"] if force_active else 0.0

        dead, dmg = body.step(f_type, intensity, cue, melt.item(), tilt.item(), fire.item())

        total_pain += dmg
        survival_time += 1

        if dead: break

    # Fitness: Survival is king.
    return survival_time

def run_universe(mode_name):
    print(f"\n🌌 UNIVERSE: {mode_name}")
    biome = get_biome(mode_name)

    parent_genes = Genotype(1.0, 1.0)

    # Pre-train a simple "Run Away" brain so they don't die of stupidity
    # Bias tilt output to be positive (Right)
    brain = Brain().to(DEVICE)
    with torch.no_grad():
        brain.fc.weight.fill_(0.0)
        brain.fc.bias[1] = 2.0 # Bias Tilt towards +1
        brain.fc.bias[2] = 2.0 # Bias Fire towards +1 (Trigger happy)

    for gen in range(GENERATIONS):
        mutants = [parent_genes.mutate() for _ in range(POP_SIZE)]
        mutants.append(parent_genes)

        scores = []
        for m in mutants:
            score = evaluate_mutant(m, biome, brain)
            scores.append((score, m))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_genes = scores[0]
        parent_genes = best_genes

        if gen % 10 == 0:
            print(f"   Gen {gen:03d} | Score: {best_score:.0f} | H: {best_genes.h_max:.2f} | B: {best_genes.b_max:.2f}")

    print(f"🏁 FINAL [{mode_name}]: H={best_genes.h_max:.2f}, B={best_genes.b_max:.2f}")

if __name__ == "__main__":
    run_universe("CRUSHER")
    run_universe("SNIPER")