import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 48
CHANNELS = 32
STEPS = 900 # Extended for 3 phases
CHECKPOINT_FILE = "rift_mnemos.pt"

print(f"🧠 PROJECT MNEMOS: MEMORY CAPACITY TEST ON {DEVICE}")

# =====================
# SUBSTRATE
# =====================
class TectonicPlate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]], dtype=torch.float32)
        self.k_stress = lap.repeat(CHANNELS,1,1,1).to(DEVICE)
        # Memory Map
        self.elasticity = nn.Parameter(torch.ones(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE) * 1.0)

    def forward(self, grid, force):
        stress = F.conv2d(grid, self.k_stress, padding=1, groups=CHANNELS)
        fracture = (stress.abs() > self.elasticity).float()

        with torch.no_grad():
            # Scarring
            self.elasticity.data += fracture * 0.5 
            # Very slow decay to allow long-term retention testing
            self.elasticity.data *= 0.9999 
            self.elasticity.data.clamp_(0.5, 15.0)

        damping = torch.where(fracture > 0, -0.2, 1.0)
        
        update = grid + (stress * -0.1) + torch.tanh(force) - (grid * 0.05)
        update = update * damping
        update = torch.clamp(update, -5.0, 5.0)

        return update, fracture

# =====================
# EGO
# =====================
class AntagonistEgo:
    def regulate(self, grid, fracture):
        f_mean = fracture.mean().item()
        if f_mean < 0.005:
            return torch.randn_like(grid) * 1.5, "PRESSURIZE", f_mean
        if f_mean > 0.15:
            return -grid * 0.8, "CONSTRAIN", f_mean
        return torch.zeros_like(grid), "OPTIMAL", f_mean

# =====================
# INPUT GENERATORS (Two-Point Strike)
# =====================
def get_hammer_force(t):
    force = torch.zeros(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE)
    
    # Only strike every 50 steps
    if t % 50 != 0:
        return force, "REST"
        
    # PHASE 1: Learn X (Steps 0-300)
    if t < 300:
        cy, cx = GRID_SIZE//4, GRID_SIZE//4 # Top-Left
        force[:,:,cy-4:cy+4,cx-4:cx+4] = 6.0
        return force, "STRIKE_X"
        
    # PHASE 2: Learn Y (Steps 300-600)
    elif t < 600:
        cy, cx = (GRID_SIZE//4)*3, (GRID_SIZE//4)*3 # Bottom-Right
        force[:,:,cy-4:cy+4,cx-4:cx+4] = 6.0
        return force, "STRIKE_Y"
        
    # PHASE 3: Recall X (Steps 600+)
    else:
        cy, cx = GRID_SIZE//4, GRID_SIZE//4 # Top-Left Again
        force[:,:,cy-4:cy+4,cx-4:cx+4] = 6.0
        return force, "RECALL_X"

# =====================
# RUNNER
# =====================
def run_mnemos():
    plate = TectonicPlate().to(DEVICE)
    ego = AntagonistEgo()
    grid = torch.randn(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE) * 0.1
    
    print("   🌱 Spawning new organism...")

    log_x = []
    log_y = []
    log_recall = []
    
    for t in range(STEPS):
        # 1. Input
        force, event_type = get_hammer_force(t)
            
        # 2. Ego
        ego_force, status, f_val = ego.regulate(grid, torch.tensor(0.0))
        
        # 3. Physics
        grid, fracture = plate(grid, force + ego_force)
        
        # 4. LOGGING (Crucial)
        if "STRIKE" in event_type or "RECALL" in event_type:
            f_mean = fracture.mean().item()
            print(f"   Step {t:03d} | Event: {event_type} | Fracture: {f_mean:.4f}")
            
            if event_type == "STRIKE_X": log_x.append(f_mean)
            if event_type == "STRIKE_Y": log_y.append(f_mean)
            if event_type == "RECALL_X": log_recall.append(f_mean)

    # ANALYSIS
    print("\n📊 ANALYSIS:")
    
    init_x = log_x[0] if log_x else 0
    final_x = log_x[-1] if log_x else 0
    final_y = log_y[-1] if log_y else 0
    recalled_x = log_recall[0] if log_recall else 0
    
    print(f"   1. Initial Trauma X: {init_x:.4f}")
    print(f"   2. Adapted Trauma X: {final_x:.4f}")
    print(f"   3. Distractor Trauma Y: {final_y:.4f}")
    print(f"   4. RECALL Trauma X: {recalled_x:.4f}")
    
    # VERDICT
    if recalled_x < (init_x * 0.5):
        print("   ✅ SUCCESS: Memory X persisted despite interference from Y.")
    else:
        print("   ⚠️ FAILURE: Catastrophic Interference (Y erased X).")
        
    if final_y < (init_x * 0.5):
         print("   ✅ SUCCESS: System learned Y independently.")
    else:
         print("   ⚠️ FAILURE: System failed to learn Y.")

if __name__ == "__main__":
    run_mnemos()