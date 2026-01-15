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
FORCE_MAG = 6.0

print(f"🧠 MNEMOS-2: MEMORY CAPACITY TEST ON {DEVICE}")

# =====================
# SUBSTRATE
# =====================
class Plate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]], dtype=torch.float32)
        self.k = lap.repeat(CH,1,1,1).to(DEVICE)
        self.elasticity = nn.Parameter(torch.ones(1,CH,GRID,GRID).to(DEVICE))

    def step(self, grid, force):
        stress = F.conv2d(grid, self.k, padding=1, groups=CH)
        fracture = (stress.abs() > self.elasticity).float()

        with torch.no_grad():
            # Scarring: Hardens heavily (0.4) to create distinct memory traces
            self.elasticity.data += fracture * 0.4
            # Decay: Very slow (0.9998) to test long-term retention
            self.elasticity.data *= 0.9998
            self.elasticity.data.clamp_(0.5, 20.0)

        damp = torch.where(fracture > 0, -0.2, 1.0)
        grid = grid + stress * -0.1 + torch.tanh(force) - grid * 0.05
        grid = grid * damp
        
        # Safety clamp
        return torch.clamp(grid, -5, 5), fracture

# =====================
# EGO
# =====================
class Ego:
    def regulate(self, grid, fracture):
        # Handle initialization case where fracture might be None or scalar
        if isinstance(fracture, float):
             f = fracture
        else:
             f = fracture.mean().item()
             
        if f < 0.005:
            return torch.randn_like(grid) * 1.2
        if f > 0.15:
            return -grid * 0.7
        return torch.zeros_like(grid)

# =====================
# STRIKE LOCATIONS
# =====================
def strike_locations(n):
    coords = []
    # Calculate spacing to fit N sites in a grid
    grid_dim = int(np.ceil(np.sqrt(n)))
    step = GRID // grid_dim
    
    idx = 0
    # Center the points in their subgrids
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
    
    # Initialize fracture for the first step
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
            
            # Apply Hammer Strike
            if t % STRIKE_INTERVAL == 0:
                force[:,:,cy-4:cy+4,cx-4:cx+4] = FORCE_MAG
                
                # ego.regulate needs the *previous* fracture state
                reg_force = ego.regulate(grid, fracture)
                
                # Step Physics
                grid, fracture = plate.step(grid, force + reg_force)
                f = fracture.mean().item()

                # Record the VERY FIRST impact on this site (Baseline Trauma)
                if first_hit is None:
                    first_hit = f
            else:
                 # Normal physics step (no hammer)
                 reg_force = ego.regulate(grid, fracture)
                 grid, fracture = plate.step(grid, reg_force)

        initial_fractures.append(first_hit)
        print(f"   -> Initial Trauma: {first_hit:.4f}")

    print("\n🔁 PHASE B — GLOBAL RECALL TEST")

    for i, (cy, cx) in enumerate(locations):
        # Single test strike
        force = torch.zeros_like(grid)
        force[:,:,cy-4:cy+4,cx-4:cx+4] = FORCE_MAG
        
        # We step once to measure reaction
        grid, fracture = plate.step(grid, force)
        f = fracture.mean().item()
        
        recall_fractures.append(f)
        
        # Compare
        baseline = initial_fractures[i]
        retention = "FORGOTTEN"
        if f < baseline * 0.5: retention = "REMEMBERED"
        if f < baseline * 0.2: retention = "STRONG"
        
        print(f"   Recall X{i+1}: {f:.4f} (Baseline: {baseline:.4f}) -> {retention}")

    # =====================
    # ANALYSIS
    # =====================
    print("\n📊 FINAL CAPACITY REPORT")

    passed = 0
    for i in range(len(initial_fractures)):
        if recall_fractures[i] < 0.5 * initial_fractures[i]:
            passed += 1

    print(f"   Sites Retained: {passed}/{MAX_SITES}")
    
    if passed >= 7:
        print("   ✅ RESULT: High Capacity Substrate.")
    elif passed >= 4:
        print("   ⚠️ RESULT: Moderate Interference.")
    else:
        print("   ❌ RESULT: Catastrophic Forgetting.")

if __name__ == "__main__":
    run()