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
GENERATIONS = 50  # Evolution cycles
POP_SIZE = 10     # Mutants per generation

print("🦠 PROJECT RIFT: POPULATION (FIXED) ON", DEVICE)
print("   (Hypothesis: Variation + Selection > Learning from Death)")

# =====================
# BIOMES
# =====================
def get_biome(mode):
    if mode == "CRUSHER": # Tank World
        return {"width": 12, "force": 8.0, "goal": "Survive Impact"}
    elif mode == "SNIPER": # Speed World
        return {"width": 2, "force": 60.0, "goal": "Dodge Lethality"}
    return None

# =====================
# GENOTYPE (The DNA)
# =====================
class Genotype:
    def __init__(self, h=1.0, b=1.0, barr=2.0):
        self.h_max = h
        self.b_max = b
        self.barrier = barr

    def mutate(self):
        # Random drift: +/- 10%
        h = self.h_max * random.uniform(0.9, 1.1)
        b = self.b_max * random.uniform(0.9, 1.1)
        return Genotype(h, b, self.barrier)

# =====================
# BODY (The Phenotype)
# =====================
class RiftBody:
    def __init__(self, genes):
        self.lap = torch.tensor([[[[0.5,1.0,0.5],[1.0,-6.0,1.0],[0.5,1.0,0.5]]]], device=DEVICE).repeat(CH,1,1,1)
        y, x = torch.meshgrid(torch.linspace(-1,1,GRID), torch.linspace(-1,1,GRID), indexing="ij")
        self.xg, self.yg = x.to(DEVICE), y.to(DEVICE)
        
        self.genes = genes # Store DNA
        self.reset()

    def reset(self):
        self.u = torch.zeros(1,CH,GRID,GRID,device=DEVICE)
        self.v = torch.zeros_like(self.u)
        self.trace = torch.zeros_like(self.u)
        self.pos = torch.tensor([0.0,0.0], device=DEVICE)
        self.vel = torch.zeros_like(self.pos)
        
        # Apply Genetics
        self.health = self.genes.h_max
        self.battery = self.genes.b_max
        self.stability = 1.0

    @torch.no_grad()
    def step(self, force, cue, melt, tilt, fire):
        self.trace = self.trace * 0.85 + cue
        
        # Stability logic
        belief = self.trace.mean().item()
        if abs(belief) > 0.1 and belief * self.pos[0].item() < 0:
            self.stability = max(0.0, self.stability - 0.04)
        
        self.battery = min(self.battery + 0.02, self.genes.b_max)

        # --- PHYSICS OF BURDEN ---
        # 1. Mass increases with Health
        mass = 1.0 + 0.5 * (self.genes.h_max - 1.0)
        # 2. Damping increases with Battery (Energy density weight)
        damping = 0.5 + 0.2 * (self.genes.b_max - 1.0) + (1.0 - self.stability)

        impulse = 0.0
        if fire > 0.5 and self.battery > 0.6 * self.genes.b_max:
            kick = (8.0 * self.battery) / math.sqrt(mass) # Heavier = Less Jump
            impulse = (1 if tilt > 0 else -1) * kick
            self.battery = 0.0

        barrier = self.genes.barrier * (1.0 - melt)
        px = self.pos[0].item()
        f_well = -4 * barrier * px * (px**2 - 0.6**2)
        f_tilt = tilt * 0.4
        
        # F = ma -> a = F/m
        accel = (f_well + f_tilt + impulse) / mass
        
        self.vel[0] += accel * DT
        self.vel *= (1.0 - damping * DT)
        self.pos = torch.clamp(self.pos + self.vel*DT, -1.0, 1.0)

        # Field
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

    def state(self):
        # FIX: Use .b_max instead of .battery_max
        return torch.tensor([
            self.trace.mean().item(), self.pos[0].item(), 
            self.battery/self.genes.b_max, self.stability, 
            self.health/self.genes.h_max, self.genes.b_max
        ], device=DEVICE)

# =====================
# BRAIN (Fixed Reflex)
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
    # Run ONE lifetime for this mutant
    body = RiftBody(genes)
    
    impact_t = 80
    cue1, cue2 = 20, 60
    
    # Left setup for consistency
    c1, c2 = 1.0, -1.0
    slice_x = (22, 26) 
    
    total_pain = 0.0
    survival_time = 0
    
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
        with torch.no_grad():
            melt, tilt, fire = brain(state)
        
        dead, dmg = body.step(force, cue, melt.item(), tilt.item(), fire.item())
        
        pain = (melt.item()**2 + fire.item()**2) * dmg * 10.0 + melt.item()*0.01
        total_pain += pain
        survival_time += 1
        
        if dead: break
        
    # Fitness Function: Maximize Survival, Minimize Pain
    fitness = survival_time - (total_pain * 5.0)
    return fitness

def run_universe(mode_name):
    print(f"\n🌌 UNIVERSE: {mode_name}")
    biome = get_biome(mode_name)
    
    # Initial Parent (Adam/Eve)
    parent_genes = Genotype(1.0, 1.0)
    
    # Brain is shared and static for now (Focus on Morphology)
    brain = Brain().to(DEVICE)
    # Pre-set a simple reflexive brain to avoid 'brain dumbness' killing the body
    with torch.no_grad():
        brain.fc.weight.fill_(0.1) 
        brain.fc.bias.fill_(0.0)

    for gen in range(GENERATIONS):
        # 1. Spawn Mutants
        mutants = [parent_genes.mutate() for _ in range(POP_SIZE)]
        mutants.append(parent_genes) # Elitism: Parent competes too
        
        # 2. Evaluate
        scores = []
        for m in mutants:
            score = evaluate_mutant(m, biome, brain)
            scores.append((score, m))
            
        # 3. Select Best
        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_genes = scores[0]
        
        # 4. Evolution Step
        parent_genes = best_genes
        
        if gen % 5 == 0:
            print(f"   Gen {gen:03d} | Best Score: {best_score:.1f} | H: {best_genes.h_max:.2f} | B: {best_genes.b_max:.2f}")

    print(f"🏁 RESULT [{mode_name}]: H={best_genes.h_max:.2f}, B={best_genes.b_max:.2f}")

if __name__ == "__main__":
    run_universe("CRUSHER") # Expect High H
    run_universe("SNIPER")  # Expect High B