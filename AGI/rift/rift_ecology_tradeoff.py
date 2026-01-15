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
STEPS = 120
EPISODES = 4000  # More episodes to allow divergence

print("⚖️ PROJECT RIFT: ECOLOGY TRADEOFF ON", DEVICE)
print("   (Hypothesis: Specialization emerges only when capabilities impose physical debts.)")

# ======================================================
# BIOMES (The Selective Pressures)
# ======================================================
BIOMES = {
    "CRUSHER": {
        "hammer_width": 12,  # Wide hammer -> Hard to dodge
        "hammer_force": 8.0, # Low damage -> Tankable
        "desc": "CRUSHER (Favors Tank)"
    },
    "SNIPER": {
        "hammer_width": 3,   # Narrow hammer -> Dodgeable
        "hammer_force": 40.0,# Instakill -> Must dodge
        "desc": "SNIPER (Favors Speed)"
    },
    "CHAOS": {
        "hammer_width": 6,
        "hammer_force": 15.0,
        "desc": "CHAOS (Favors Balance)"
    }
}

# ======================================================
# BODY (Physics with Morphological Debt)
# ======================================================
class RiftTradeoffBody:
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
        self.xg = x.to(DEVICE)
        self.yg = y.to(DEVICE)

        # Genotype (Evolvable Traits)
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

        # Phenotype (Current State)
        self.health = self.health_max
        self.battery = self.battery_max * 0.5
        self.stability = 1.0

        self.used_impulse = False
        self.survived_stress = False

    @torch.no_grad()
    def step(self, force, cue, melt, tilt, fire):
        # 1. TRACE
        self.trace = self.trace * 0.85 + cue
        belief = self.trace.mean().item()
        reality = self.pos[0].item()

        # 2. DISSONANCE
        if abs(belief) > 0.1 and belief * reality < 0:
            self.stability -= 0.04
            self.survived_stress = True
        
        self.stability = max(0.0, min(1.0, self.stability))

        # 3. BATTERY
        self.battery = min(self.battery + 0.02, self.battery_max)

        # ==========================================
        # 🔥 THE LAWS OF MORPHOLOGICAL DEBT
        # ==========================================
        
        # Law 1: Battery Mass (Capacity adds Damping)
        # More fuel = Heavier body = Harder to maintain velocity
        debt_damping = 0.5 + 0.3 * (self.battery_max - 1.0)

        # Law 2: Health Inertia (Armor adds Mass)
        # More health = Higher mass = Harder to accelerate (F=ma)
        debt_mass = 1.0 + 0.5 * (self.health_max - 1.0)

        # Law 3: Stability Stiffness (Calmness reduces Reflex Speed)
        # High stability = Low gain (Stubborn)
        reflex_gain = 0.4 / (1.0 + 0.5 * self.stability)

        # ==========================================

        impulse = 0.0
        # Fire logic
        if fire > 0.5 and self.battery > 0.6 * self.battery_max:
            # Impulse force must overcome Mass Debt
            impulse = (1 if tilt > 0 else -1) * self.battery * (6.0 / debt_mass)
            self.battery = 0.0
            self.used_impulse = True

        # Landscape Forces
        barrier = self.barrier_base * (1.0 - melt)
        px = self.pos[0].item()
        
        # Well Force
        f_well = -4 * barrier * px * (px**2 - 0.6**2) 
        
        # Tilt Force (affected by reflex gain)
        f_tilt = tilt * reflex_gain
        
        total_force = f_well + f_tilt + impulse
        
        # Newton's Second Law with Debt: a = F / m
        accel_x = total_force / debt_mass
        accel_y = (-2 * self.pos[1].item()) / debt_mass

        # Integration
        self.vel += torch.tensor([accel_x, accel_y], device=DEVICE) * DT
        self.vel *= (1.0 - debt_damping * DT) # Damping applied to velocity
        
        self.pos = torch.clamp(self.pos + self.vel*DT, -1.0, 1.0)

        # Field Physics
        dist = (self.xg - self.pos[0])**2 + (self.yg - self.pos[1])**2
        shield = torch.exp(-dist/(2*0.4**2)).unsqueeze(0).unsqueeze(0).expand(1,CH,GRID,GRID)

        eff = force * (1.0 - shield)
        diff = F.conv2d(self.u, self.lap, padding=1, groups=CH)
        self.v += (diff - self.u + eff - 0.1*self.v)*DT
        self.u += self.v*DT

        # Damage
        dmg = 0.0
        deform = self.u.abs().max().item()
        if deform > 0.1:
            dmg = deform * 0.4
            self.health -= dmg

        return self.health <= 0, dmg

    def evolve(self):
        # Evolution is now risky. Growing increases Debt.
        
        # If impulse was useful, grow battery (but accept damping debt)
        if self.used_impulse:
            self.battery_max += 0.05
            
        # If stressed but alive, grow health (but accept mass debt)
        if self.survived_stress and self.health > 0:
            self.health_max += 0.05
            
        # Atrophy: If unused, shrink slightly to reduce debt
        if not self.used_impulse:
            self.battery_max = max(1.0, self.battery_max * 0.995)
            
        # Death Penalty
        if self.health <= 0:
            self.barrier_base = max(1.0, self.barrier_base * 0.95)

    def state(self):
        return torch.tensor([
            self.trace.mean().item(),
            self.pos[0].item(),
            self.battery/self.battery_max,
            self.stability,
            self.health/self.health_max,
            self.battery_max
        ], device=DEVICE)

# ======================================================
# BRAIN
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
    body = RiftTradeoffBody()
    brain = Brain().to(DEVICE)
    opt = optim.Adam(brain.parameters(),lr=0.01)

    biome_names = list(BIOMES.keys())

    # History tracking
    history = {"CRUSHER": [], "SNIPER": [], "CHAOS": []}

    print("\n⚔️ BEGINNING EVOLUTIONARY RUN...")

    for ep in range(EPISODES):
        body.reset()
        
        # Pick Biome
        b_name = random.choice(biome_names)
        biome = BIOMES[b_name]

        impact_t = random.randint(80,100)
        cue1, cue2 = impact_t-60, impact_t-20

        # Randomize Left/Right
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
                w = biome["hammer_width"] // 2
                # Center of impact
                cx = (slice_x[0] + slice_x[1]) // 2
                # Bounds check
                x1 = max(0, cx - w)
                x2 = min(32, cx + w)
                force[:,:,14:18, x1:x2] = biome["hammer_force"]

            state = body.state()
            melt, tilt, fire = brain(state)

            dead, dmg = body.step(
                force, cue, 
                melt.item(), tilt.item(), fire.item()
            )

            # Hebbian-ish Loss
            pain = (melt**2 + tilt**2 + fire**2) * (dmg * 10.0) 
            pain += melt*0.01 + torch.abs(tilt)*0.005 + fire*0.005

            opt.zero_grad()
            pain.backward()
            opt.step()

            if dead:
                break

        body.evolve()
        
        # Log stats for this biome
        history[b_name].append((body.health_max, body.battery_max))

        if ep % 200 == 0:
            # Calculate averages for recent history
            print(f"--- EP {ep:04d} ---")
            for b in biome_names:
                recs = history[b][-50:] # Last 50 runs in this biome
                if not recs: continue
                avg_h = sum(r[0] for r in recs)/len(recs)
                avg_b = sum(r[1] for r in recs)/len(recs)
                print(f"   [{b[:4]}] H_max: {avg_h:.2f} | B_max: {avg_b:.2f}")

if __name__ == "__main__":
    run()