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
STEPS = 600
CHECKPOINT_FILE = "rift_core_memory.pt"

print(f"🌋 RIFT v4 EXPERIMENT: EPISODIC TRAUMA TEST ON {DEVICE}")

# =====================
# SUBSTRATE
# =====================
class TectonicPlate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]], dtype=torch.float32)
        self.k_stress = lap.repeat(CHANNELS,1,1,1).to(DEVICE)
        
        # Elasticity Map (Memory)
        self.elasticity = nn.Parameter(torch.ones(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE) * 1.0)

    def forward(self, grid, force, mode):
        # 1. Stress
        stress = F.conv2d(grid, self.k_stress, padding=1, groups=CHANNELS)
        
        # 2. Fracture Check
        fracture = (stress.abs() > self.elasticity).float()

        # 3. Learning (Scarring)
        with torch.no_grad():
            # HARDENING: Only happens if there is fracture
            self.elasticity.data += fracture * 0.5  # Stronger scarring (0.2 -> 0.5)
            
            # DECAY: Controlled by Mode
            decay_rate = 0.9999 if mode == 'B' else 0.995 # Run B remembers almost everything
            self.elasticity.data *= decay_rate
            self.elasticity.data.clamp_(0.5, 15.0)

        # 4. Physics
        # Damping: Fracture absorbs energy
        damping = torch.where(fracture > 0, -0.2, 1.0)
        
        # Update
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
        
        # Homeostasis
        if f_mean < 0.005:
            return torch.randn_like(grid) * 1.5, "PRESSURIZE", f_mean
        if f_mean > 0.15:
            return -grid * 0.8, "CONSTRAIN", f_mean
            
        return torch.zeros_like(grid), "OPTIMAL", f_mean

# =====================
# INPUT GENERATORS
# =====================
def input_baseline(t):
    return torch.randn(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE) * 0.05

def input_shock_hammer(t):
    # Run B/C: Single-step massive shock
    force = torch.zeros(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE)
    if t % 50 == 0: # THE HAMMER STRIKE
        cy, cx = GRID_SIZE//2, GRID_SIZE//2
        force[:,:,cy-4:cy+4,cx-4:cx+4] = 6.0 # Massive force
    return force

# =====================
# MUTATION (Run C)
# =====================
def mutate_topology(plate):
    with torch.no_grad():
        mask_weak = plate.elasticity.data < 1.0
        plate.elasticity.data[mask_weak] *= 0.8 # Weaken the weak (Differentiation)
        plate.elasticity.data[~mask_weak] *= 1.2 # Strengthen the strong
        plate.elasticity.data.clamp_(0.1, 20.0)

# =====================
# RUNNER
# =====================
def run_experiment(mode):
    print(f"\n🧪 STARTING RUN: {mode}")
    
    plate = TectonicPlate().to(DEVICE)
    ego = AntagonistEgo()
    grid = torch.randn(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE) * 0.1
    
    # Load State
    if os.path.exists(CHECKPOINT_FILE):
        print(f"   📂 Loading organism from {CHECKPOINT_FILE}...")
        state = torch.load(CHECKPOINT_FILE, weights_only=False)
        plate.load_state_dict(state['plate'])
        grid = state['grid']
    else:
        print("   🌱 Spawning new organism...")

    log_impacts = []
    
    for t in range(STEPS):
        # 1. Environment
        if mode == 'A':
            force = input_baseline(t)
        else:
            force = input_shock_hammer(t)
            
        # 2. Ego
        ego_force, status, f_val = ego.regulate(grid, torch.tensor(0.0))
        
        # 3. Physics
        grid, fracture = plate(grid, force + ego_force, mode)
        
        # 4. Mutation
        if mode == 'C' and t % 100 == 0:
            mutate_topology(plate)
            status = "MUTATION"
            
        # 5. LOGGING (The Important Part)
        if mode in ['B', 'C'] and t % 50 == 0:
            # We only log the IMPACT MOMENT
            f_mean = fracture.mean().item()
            log_impacts.append(f_mean)
            print(f"   Step {t:03d} | IMPACT Fracture: {f_mean:.4f} | Status: TRAUMA EVENT")
        elif t % 50 == 0:
             f_mean = fracture.mean().item()
             print(f"   Step {t:03d} | Fracture: {f_mean:.4f} | Status: {status}")

    # Save
    torch.save({'plate': plate.state_dict(), 'grid': grid}, CHECKPOINT_FILE)
    print(f"   💾 Checkpoint saved.")
    
    if mode in ['B', 'C']:
        print(f"   📊 IMPACT LOG: {log_impacts}")
        if log_impacts[0] > log_impacts[-1]:
            print("   ✅ SUCCESS: System adapted to stress (Fracture decreased).")
        else:
            print("   ⚠️ FAILURE: System did not adapt.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["A", "B", "C"], required=True)
    args = parser.parse_args()
    run_experiment(args.mode)