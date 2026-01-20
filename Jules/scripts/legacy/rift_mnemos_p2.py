import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 64
CH = 16
DT = 0.05
STEPS_TRAIN = 800
STEPS_TEST = 200

print(f"🧠 MNEMOS-P4: PURE WAVE PREDICTION TEST ON {DEVICE}")
print("   (Hypothesis: Eroded paths allow Cue to pre-load Impact site, altering damage.)")

# =====================
# SUBSTRATE
# =====================
class Substrate(nn.Module):
    def __init__(self):
        super().__init__()
        # Laplacian
        lap = torch.tensor([[[[0.5, 1.0, 0.5],
                              [1.0, -6.0, 1.0],
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        # STATE
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)

        # PARAMETERS
        # Structure: Resistance to breaking (Memory)
        self.k_structure = nn.Parameter(torch.ones(1, CH, GRID, GRID, device=DEVICE) * 0.1)

        # Conductivity: Ease of wave travel (Association)
        # Starts LOW to prove learning creates the path
        self.conductivity = nn.Parameter(torch.ones(1, CH, GRID, GRID, device=DEVICE) * 0.05)

    def step(self, external_force, mode="learn"):
        # WAVE PHYSICS
        # Stress Flow = Conductivity * Laplacian(u)
        diffusion = self.conductivity * F.conv2d(self.u, self.laplacian, padding=1, groups=CH)

        # Restoring Force = -K * u
        restoring = -self.k_structure * self.u

        # Acceleration
        accel = diffusion + restoring + external_force - (0.05 * self.v) # Low Damping

        # Integration
        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # ENERGY / VIBRATION
        vibration = self.u.abs()

        # LEARNING (Hebbian Erosion + Scarring)
        with torch.no_grad():
            if mode == "learn":
                # 1. SCARRING (Trauma -> Hardening)
                # High vibration creates Structure (Callus)
                fracture = (vibration > 1.5).float()
                self.k_structure.data += fracture * 0.05
                self.k_structure.data.clamp_(0.1, 8.0)

                # 2. HEBBIAN EROSION (Co-activity -> Conductivity)
                # If point A and neighbor B vibrate together, erode the barrier
                neighbor_activity = F.avg_pool2d(vibration, 3, 1, 1)
                coincidence = vibration * neighbor_activity

                # Erode (Increase Conductivity)
                self.conductivity.data += coincidence * 0.005
                self.conductivity.data.clamp_(0.05, 4.0) # Max limit prevents explosion

        # MEASURE DAMAGE (Simulated Fracture Event)
        # We don't verify fracture logic here to avoid feedback loops in testing,
        # we just measure "Potential Fracture" to see if stress is reduced.
        stress_load = (self.k_structure * self.u).abs()
        fracture_mass = (stress_load > 2.0).float().sum().item()

        return self.u, fracture_mass

# =====================
# RUNNER
# =====================
def run():
    system = Substrate().to(DEVICE)

    cy, cx = GRID // 2, GRID // 2
    cue_loc = (cy, cx - 12)  # Far left
    imp_loc = (cy, cx)       # Center

    print("\n🔨 PHASE 1: TRAINING (Creating the Path)")

    for t in range(STEPS_TRAIN):
        force = torch.zeros_like(system.u)

        # CUE at t=10
        if t % 80 == 10:
            force[:,:,cue_loc[0]-2:cue_loc[0]+2, cue_loc[1]-2:cue_loc[1]+2] = 5.0

        # IMPACT at t=35 (Gap allows wave travel)
        if t % 80 == 35:
            force[:,:,imp_loc[0]-2:imp_loc[0]+2, imp_loc[1]-2:imp_loc[1]+2] = 15.0

        u, damage = system.step(force, mode="learn")

        if t % 160 == 0:
            path_quality = system.conductivity[:,:,cy, cx-6].mean().item()
            print(f"   Step {t:03d} | Path Conductivity: {path_quality:.4f} | Training Damage: {damage}")

    # RESET STATE (But keep Memory/Conductivity)
    system.u.zero_()
    system.v.zero_()

    print("\n🔮 PHASE 2: THE TEST (Cued vs Uncued)")

    # TEST A: UNCUED IMPACT (Baseline)
    print("   Running Control (No Cue)...")
    damage_uncued = 0
    for t in range(100):
        force = torch.zeros_like(system.u)
        # No Cue
        if t == 35: # Same impact time
             force[:,:,imp_loc[0]-2:imp_loc[0]+2, imp_loc[1]-2:imp_loc[1]+2] = 15.0

        u, d = system.step(force, mode="test")
        damage_uncued += d

    # RESET STATE
    system.u.zero_()
    system.v.zero_()

    # TEST B: CUED IMPACT
    print("   Running Test (With Cue)...")
    damage_cued = 0
    for t in range(100):
        force = torch.zeros_like(system.u)
        # Cue at t=10
        if t == 10:
             force[:,:,cue_loc[0]-2:cue_loc[0]+2, cue_loc[1]-2:cue_loc[1]+2] = 5.0
        # Impact at t=35
        if t == 35:
             force[:,:,imp_loc[0]-2:imp_loc[0]+2, imp_loc[1]-2:imp_loc[1]+2] = 15.0

        u, d = system.step(force, mode="test")
        damage_cued += d

    print("\n📊 PREDICTION REPORT")
    print(f"   Total Damage (Uncued): {damage_uncued:.1f}")
    print(f"   Total Damage (Cued):   {damage_cued:.1f}")

    delta = damage_uncued - damage_cued
    pct = (delta / damage_uncued) * 100 if damage_uncued > 0 else 0

    print(f"   Damage Reduction: {delta:.1f} ({pct:.1f}%)")

    if delta > 0:
        print("   ✅ SUCCESS: Cue signal physically mitigated the impact damage.")
        print("   (Mechanism: Pre-arrival of cue wave altered local stress state.)")
    elif delta < 0:
        print("   ⚠️ SENSITIZATION: Cue signal made impact WORSE (Constructive Interference).")
    else:
        print("   ❌ FAILURE: Cue had no effect (Signal didn't arrive).")

if __name__ == "__main__":
    run()