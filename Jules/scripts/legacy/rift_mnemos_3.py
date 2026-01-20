import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 48
CH = 32
STEPS_PER_SITE = 200
STRIKE_INTERVAL = 20
MAX_SITES = 9
FORCE_MAG = 8.0 # Stronger Hammer

print(f"🧠 MNEMOS-3: TUNED CAPACITY TEST ON {DEVICE}")

# =====================
# SUBSTRATE
# =====================
class Plate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]], dtype=torch.float32)
        self.k = lap.repeat(CH,1,1,1).to(DEVICE)
        self.elasticity = nn.Parameter(torch.ones(1,CH,GRID,GRID).to(DEVICE) * 1.0)

    def step(self, grid, force):
        stress = F.conv2d(grid, self.k, padding=1, groups=CH)

        # Fracture Check
        fracture = (stress.abs() > self.elasticity).float()

        with torch.no_grad():
            # Scarring:
            # Only harden if fracture happened.
            self.elasticity.data += fracture * 0.5

            # Decay:
            self.elasticity.data *= 0.9998
            self.elasticity.data.clamp_(0.5, 20.0)

        damp = torch.where(fracture > 0, -0.2, 1.0)
        grid = grid + stress * -0.1 + torch.tanh(force) - grid * 0.05
        grid = grid * damp

        return torch.clamp(grid, -5, 5), fracture

# =====================
# EGO (TUNED)
# =====================
class Ego:
    def regulate(self, grid, fracture):
        if isinstance(fracture, float): f = fracture
        else: f = fracture.mean().item()

        if f < 0.005:
            # REDUCED KICK: 0.5 is less than initial elasticity (1.0)
            # This prevents the Ego from creating scars during "Boredom"
            return torch.randn_like(grid) * 0.5

        if f > 0.15:
            return -grid * 0.7
        return torch.zeros_like(grid)

# =====================
# STRIKE LOCATIONS
# =====================
def strike_locations(n):
    coords = []
    grid_dim = int(np.ceil(np.sqrt(n)))
    step = GRID // grid_dim
    idx = 0
    offset = step // 2
    for y in range(offset, GRID, step):
        for x in range(offset, GRID, step):
            if idx < n:
                coords.append((y, x))
                idx += 1
    return coords

# =====================
# RUNNER
# =====================
def run():
    plate = Plate().to(DEVICE)
    ego = Ego()
    grid = torch.randn(1,CH,GRID,GRID).to(DEVICE) * 0.1

    fracture = torch.zeros_like(grid)
    locations = strike_locations(MAX_SITES)
    initial_fractures = []
    recall_fractures = []

    print(f"\n🔨 PHASE A — SEQUENTIAL LEARNING ({MAX_SITES} SITES)")

    for i, (cy, cx) in enumerate(locations):
        print(f"\n   Targeting Site X{i+1} at ({cy},{cx})...")
        first_hit = None

        for t in range(STEPS_PER_SITE):
            force = torch.zeros_like(grid)

            if t % STRIKE_INTERVAL == 0:
                force[:,:,cy-4:cy+4,cx-4:cx+4] = FORCE_MAG

                reg_force = ego.regulate(grid, fracture)
                grid, fracture = plate.step(grid, force + reg_force)
                f = fracture.mean().item()

                if first_hit is None:
                    first_hit = f
            else:
                 reg_force = ego.regulate(grid, fracture)
                 grid, fracture = plate.step(grid, reg_force)

        # Handle the "Callus" case (If first hit was 0.0)
        if first_hit < 0.0001:
            print(f"   -> WARNING: Site was already hardened (Global Callus).")
        else:
            print(f"   -> Initial Trauma: {first_hit:.4f}")
        initial_fractures.append(first_hit)

    print("\n🔁 PHASE B — GLOBAL RECALL TEST")

    passed = 0
    for i, (cy, cx) in enumerate(locations):
        force = torch.zeros_like(grid)
        force[:,:,cy-4:cy+4,cx-4:cx+4] = FORCE_MAG

        grid, fracture = plate.step(grid, force)
        f = fracture.mean().item()

        recall_fractures.append(f)
        baseline = initial_fractures[i]

        status = "FAIL"
        # If baseline was valid (trauma occurred) and recall is low -> REMEMBERED
        if baseline > 0.001 and f < baseline * 0.5:
            status = "REMEMBERED"
            passed += 1
        # If baseline was 0 (Callus) and recall is 0 -> BLOCKED (Not memory)
        elif baseline <= 0.001 and f <= 0.001:
            status = "BLOCKED (Pre-Hardened)"

        print(f"   Recall X{i+1}: {f:.4f} (Baseline: {baseline:.4f}) -> {status}")

    # =====================
    # ANALYSIS
    # =====================
    print("\n📊 CAPACITY REPORT")
    print(f"   Sites Retained: {passed}/{MAX_SITES}")

    if passed >= 7:
        print("   ✅ RESULT: High Capacity.")
    elif passed >= 4:
        print("   ⚠️ RESULT: Moderate Interference.")
    else:
        print("   ❌ RESULT: Ego Dominance / Forgetting.")

if __name__ == "__main__":
    run()