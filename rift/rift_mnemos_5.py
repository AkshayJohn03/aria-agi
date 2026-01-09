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

print(f"🧠 MNEMOS-5: SPATIAL GATING TEST ON {DEVICE}")

# =====================
# SUBSTRATE
# =====================
class Plate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]], dtype=torch.float32)
        self.k_stress = lap.repeat(CH,1,1,1).to(DEVICE)
        
        # Two Layers
        self.elasticity_fast = nn.Parameter(torch.ones(1,CH,GRID,GRID).to(DEVICE) * 0.5)
        self.elasticity_slow = nn.Parameter(torch.zeros(1,CH,GRID,GRID).to(DEVICE))

    def step(self, grid, force):
        # 1. Calculate Stress
        stress = F.conv2d(grid, self.k_stress, padding=1, groups=CH)
        
        # 2. Total Resistance (Skin + Bone)
        total_elasticity = self.elasticity_fast + self.elasticity_slow
        
        # 3. Fracture (Physical Breakdown)
        fracture = (stress.abs() > total_elasticity).float()

        # 4. SPATIAL GATING (The Novelty Filter)
        # We calculate the "Sharpness" of the stress field
        # High Sharpness = Hammer (Signal). Low Sharpness = Ego (Noise).
        avg_stress = F.avg_pool2d(stress.abs(), kernel_size=3, stride=1, padding=1)
        sharpness = (stress.abs() - avg_stress).abs()
        
        # Gate: Only update SLOW memory if the event is "Sharp" (Structured)
        # Threshold chosen empirically to separate Hammer from Noise
        memory_gate = (sharpness > 0.5).float()

        with torch.no_grad():
            # FAST LAYER (Skin): Reacts to everything (Ego + Hammer) to keep system alive
            self.elasticity_fast.data += fracture * 0.5
            self.elasticity_fast.data *= 0.90 # Fast decay (sheds the callus quickly)
            self.elasticity_fast.data.clamp_(0.5, 5.0)
            
            # SLOW LAYER (Bone): Reacts ONLY to Sharp Events (Gated)
            # We use the GATE, not just the fracture
            learning_signal = fracture * memory_gate
            
            self.elasticity_slow.data += learning_signal * 0.5
            self.elasticity_slow.data *= 0.9999 # Permanent
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
            # Noise is diffuse, shouldn't trigger the Spatial Gate
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
                if first_hit is None: first_hit = f
            else:
                 reg_force = ego.regulate(grid, fracture)
                 grid, fracture = plate.step(grid, reg_force)

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
            status = "BLOCKED"
        
        print(f"   Recall X{i+1}: {f:.4f} (Baseline: {baseline:.4f}) -> {status}")

    print("\n📊 SPATIAL GATING REPORT")
    print(f"   Sites Retained: {passed}/{MAX_SITES}")
    
    if passed >= 7:
        print("   ✅ RESULT: Spatial Gating Solved Interference.")
    else:
        print("   ❌ RESULT: Substrate Limits Reached.")

if __name__ == "__main__":
    run()