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
FORCE_MAG = 8.0

print(f"🧠 MNEMOS-4: DUAL-TIMESCALE MEMORY TEST ON {DEVICE}")

# =====================
# SUBSTRATE (Dual Elasticity)
# =====================
class Plate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]], dtype=torch.float32)
        self.k = lap.repeat(CH,1,1,1).to(DEVICE)

        # TWO MEMORY SYSTEMS
        # 1. Fast (Skin): Handles daily stress/Ego. Forgets fast.
        self.elasticity_fast = nn.Parameter(torch.ones(1,CH,GRID,GRID).to(DEVICE) * 0.5)
        # 2. Slow (Bone): Handles trauma. Forgets never.
        self.elasticity_slow = nn.Parameter(torch.zeros(1,CH,GRID,GRID).to(DEVICE))

    def step(self, grid, force):
        stress = F.conv2d(grid, self.k, padding=1, groups=CH)

        # Total Resistance
        total_elasticity = self.elasticity_fast + self.elasticity_slow

        # Fracture Check
        fracture = (stress.abs() > total_elasticity).float()

        with torch.no_grad():
            # FAST LEARNING (The Skin)
            # High rate (0.5), High decay (0.95)
            # This absorbs the Ego's noise so it doesn't scar the bone
            self.elasticity_fast.data += fracture * 0.5
            self.elasticity_fast.data *= 0.95
            self.elasticity_fast.data.clamp_(0.5, 5.0)

            # SLOW LEARNING (The Bone)
            # Low rate (0.2), Low decay (0.9999)
            # Only hardens if fracture persists despite skin response
            self.elasticity_slow.data += fracture * 0.2
            self.elasticity_slow.data *= 0.9999
            self.elasticity_slow.data.clamp_(0.0, 15.0)

        damp = torch.where(fracture > 0, -0.2, 1.0)
        grid = grid + stress * -0.1 + torch.tanh(force) - grid * 0.05
        grid = grid * damp

        return torch.clamp(grid, -5, 5), fracture

# =====================
# EGO
# =====================
class Ego:
    def regulate(self, grid, fracture):
        if isinstance(fracture, float): f = fracture
        else: f = fracture.mean().item()

        if f < 0.005:
            # Kick should be absorbed by Fast layer
            return torch.randn_like(grid) * 0.8
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

        # Baseline Check
        if first_hit < 0.0001:
            print(f"   -> WARNING: Site pre-hardened.")
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
        if baseline > 0.001 and f < baseline * 0.5:
            status = "REMEMBERED"
            passed += 1
        elif baseline <= 0.001 and f <= 0.001:
            status = "BLOCKED (Pre-Hardened)"

        print(f"   Recall X{i+1}: {f:.4f} (Baseline: {baseline:.4f}) -> {status}")

    print("\n📊 DUAL-LAYER CAPACITY REPORT")
    print(f"   Sites Retained: {passed}/{MAX_SITES}")

    if passed >= 7:
        print("   ✅ RESULT: Timescale Separation Solved Interference.")
    else:
        print("   ❌ RESULT: Still Interfering.")

if __name__ == "__main__":
    run()