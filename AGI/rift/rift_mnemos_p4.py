import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 64
CH = 16
DT = 0.05
STEPS_TRAIN = 1000
STEPS_TEST = 200

print(f"🧠 MNEMOS-P6: AMPLIFIED HEBBIAN TEST ON {DEVICE}")

# =====================
# SUBSTRATE
# =====================
class Substrate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0.5, 1.0, 0.5], 
                              [1.0, -6.0, 1.0], 
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        
        # PARAMETERS
        # Structure: Softer (0.5) to allow vibration
        self.k_structure = nn.Parameter(torch.ones(1, CH, GRID, GRID, device=DEVICE) * 0.5)
        
        # Conductivity: Higher Base (0.05) to allow signal travel
        self.conductivity = nn.Parameter(torch.ones(1, CH, GRID, GRID, device=DEVICE) * 0.05)

    def step(self, external_force, mode="learn"):
        # PHYSICS
        diffusion = self.conductivity * F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -self.k_structure * self.u
        accel = diffusion + restoring + external_force - (0.05 * self.v)
        
        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT
        
        # ENERGY
        stress_load = (self.k_structure * self.u).abs()
        vibration = self.u.abs()
        
        # LEARNING
        with torch.no_grad():
            if mode == "learn":
                # 1. SCARRING (Hardening)
                fracture = (stress_load > 0.8).float()
                self.k_structure.data += fracture * 0.05
                self.k_structure.data.clamp_(0.5, 5.0)
                
                # 2. HEBBIAN EROSION (Aggressive)
                # We need the wave to bridge the gap.
                neighbor_activity = F.avg_pool2d(vibration, 7, 1, 3) # Wide reach
                coincidence = vibration * neighbor_activity
                
                # AMPLIFIED LEARNING RATE (x10)
                self.conductivity.data += coincidence * 0.1 
                self.conductivity.data.clamp_(0.05, 5.0)
                
        return self.u, stress_load

# =====================
# RUNNER
# =====================
def run():
    system = Substrate().to(DEVICE)
    
    cy, cx = GRID // 2, GRID // 2
    cue_loc = (cy, cx - 12)  
    imp_loc = (cy, cx)       
    
    print("\n🔨 PHASE 1: TRAINING (Building the Wire)")
    
    for t in range(STEPS_TRAIN):
        force = torch.zeros_like(system.u)
        
        # CUE at t=10
        if t % 80 == 10:
            force[:,:,cue_loc[0]-2:cue_loc[0]+2, cue_loc[1]-2:cue_loc[1]+2] = 8.0
            
        # IMPACT at t=35
        if t % 80 == 35:
            force[:,:,imp_loc[0]-2:imp_loc[0]+2, imp_loc[1]-2:imp_loc[1]+2] = 12.0
            
        u, stress = system.step(force, mode="learn")
        
        if t % 200 == 0:
            # Check conductivity in the middle of the path
            mid_y, mid_x = cy, cx - 6
            path_quality = system.conductivity[:,:,mid_y, mid_x].mean().item()
            print(f"   Step {t:04d} | Path Conductivity: {path_quality:.4f}")

    # RESET STATE (Keep Memory)
    system.u.zero_()
    system.v.zero_()
    
    print("\n🔮 PHASE 2: THE DISSIPATION TEST")
    
    # TEST A: UNCUED
    print("   Running Control (No Cue)...")
    peak_stress_uncued = 0
    for t in range(100):
        force = torch.zeros_like(system.u)
        if t == 35: force[:,:,imp_loc[0]-2:imp_loc[0]+2, imp_loc[1]-2:imp_loc[1]+2] = 12.0
        
        u, stress = system.step(force, mode="test")
        
        local_stress = stress[:,:,imp_loc[0]-2:imp_loc[0]+2, imp_loc[1]-2:imp_loc[1]+2].max().item()
        if local_stress > peak_stress_uncued: peak_stress_uncued = local_stress
        
    system.u.zero_()
    system.v.zero_()
    
    # TEST B: CUED
    print("   Running Test (With Cue)...")
    peak_stress_cued = 0
    for t in range(100):
        force = torch.zeros_like(system.u)
        if t == 10: force[:,:,cue_loc[0]-2:cue_loc[0]+2, cue_loc[1]-2:cue_loc[1]+2] = 8.0
        if t == 35: force[:,:,imp_loc[0]-2:imp_loc[0]+2, imp_loc[1]-2:imp_loc[1]+2] = 12.0
        
        u, stress = system.step(force, mode="test")
        
        local_stress = stress[:,:,imp_loc[0]-2:imp_loc[0]+2, imp_loc[1]-2:imp_loc[1]+2].max().item()
        if local_stress > peak_stress_cued: peak_stress_cued = local_stress

    print("\n📊 FINAL REPORT")
    print(f"   Peak Stress (Uncued): {peak_stress_uncued:.4f}")
    print(f"   Peak Stress (Cued):   {peak_stress_cued:.4f}")
    
    delta = peak_stress_uncued - peak_stress_cued
    pct = (delta / peak_stress_uncued) * 100
    
    print(f"   Dissipation Gain: {pct:.2f}%")
    
    if pct > 1.0:
        print("   ✅ SUCCESS: Learned path dissipated trauma energy.")
    else:
        print("   ❌ FAILURE: No predictive dissipation.")

if __name__ == "__main__":
    run()