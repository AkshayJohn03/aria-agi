import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt

# =====================
# CONFIG & PHYSICS
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 48
CHANNELS = 32
STEPS = 600
CHECKPOINT_FILE = "rift_core_memory.pt"

print(f"🌋 RIFT EXPERIMENT HARNESS ON {DEVICE}")

# =====================
# SUBSTRATE (The Tectonic Plate)
# =====================
class TectonicPlate(nn.Module):
    def __init__(self):
        super().__init__()
        # Laplacian Kernel (Stress Detector)
        lap = torch.tensor([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]], dtype=torch.float32)
        self.k_stress = lap.repeat(CHANNELS,1,1,1).to(DEVICE)

        # Elasticity Map (The Memory)
        # Starts uniform (1.0). Hardens with trauma. Decays with entropy.
        self.elasticity = nn.Parameter(torch.ones(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE) * 1.0)

    def forward(self, grid, force):
        # 1. Calc Stress
        stress = F.conv2d(grid, self.k_stress, padding=1, groups=CHANNELS)

        # 2. Check Fracture
        fracture = (stress.abs() > self.elasticity).float()

        # 3. Learning (Scarring + Decay)
        with torch.no_grad():
            # Trauma hardens the location (Long-term Potentiation)
            self.elasticity += fracture * 0.2
            # Entropy softens everything (Forgetting curve)
            self.elasticity *= 0.998
            self.elasticity.clamp_(0.5, 10.0)

        # 4. Physics Update
        # Fracture reverses local velocity (Energy Dissipation)
        damping = torch.where(fracture > 0, -0.5, 1.0)

        update = grid + (stress * -0.1) + torch.tanh(force) - (grid * 0.05)
        update = update * damping
        update = torch.clamp(update, -5.0, 5.0)

        return update, fracture

# =====================
# EGO (The Regulator)
# =====================
class AntagonistEgo:
    def regulate(self, grid, fracture):
        f_mean = fracture.mean().item()

        # Homeostatic Bounds
        if f_mean < 0.001:
            # Boreom -> Kick
            return torch.randn_like(grid) * 1.5, "PRESSURIZE", f_mean
        if f_mean > 0.10:
            # Panic -> Damp
            return -grid * 0.8, "CONSTRAIN", f_mean

        return torch.zeros_like(grid), "OPTIMAL", f_mean

# =====================
# INPUT GENERATORS (The Environment)
# =====================
def input_baseline(t):
    # Run A: Just background noise. Existence test.
    return torch.randn(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE) * 0.05

def input_stress_pattern(t):
    # Run B/C: A specific "Trauma Event" repeats every 50 steps.
    # Does the system learn to anticipate/harden against THIS specific pattern?
    force = torch.zeros(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE)

    if t % 50 < 10: # The Event lasts 10 steps
        # A hard block of force in the center
        cy, cx = GRID_SIZE//2, GRID_SIZE//2
        force[:,:,cy-4:cy+4,cx-4:cx+4] = 2.0

    return force

# =====================
# TOPOLOGY MUTATION (Run C Only)
# =====================
def mutate_topology(plate):
    # Irreversible structural change
    with torch.no_grad():
        # Areas that are weak (low elasticity) get weaker (Split)
        # Areas that are strong get stronger (Merge)
        mask_weak = plate.elasticity < 1.0
        plate.elasticity[mask_weak] *= 0.9
        plate.elasticity[~mask_weak] *= 1.1
        plate.elasticity.clamp_(0.1, 15.0)

# =====================
# MAIN RUNNER
# =====================
def run_experiment(mode):
    print(f"\n🧪 STARTING RUN: {mode}")

    plate = TectonicPlate().to(DEVICE)
    ego = AntagonistEgo()
    grid = torch.randn(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE) * 0.1

    # Load previous state if exists (Continuity of Identity)
    if os.path.exists(CHECKPOINT_FILE):
        print(f"   📂 Loading organism from {CHECKPOINT_FILE}...")
        state = torch.load(CHECKPOINT_FILE)
        plate.load_state_dict(state['plate'])
        grid = state['grid']
    else:
        print("   🌱 Spawning new organism...")

    log_fracture = []

    for t in range(STEPS):
        # 1. Environment
        if mode == 'A':
            force = input_baseline(t)
        else:
            force = input_stress_pattern(t)

        # 2. Ego Regulation
        ego_force, status, f_val = ego.regulate(grid, torch.tensor(0.0)) # reactive approx

        # 3. Physics
        grid, fracture = plate(grid, force + ego_force)

        # 4. Mutation (Run C Only)
        if mode == 'C' and t % 100 == 0:
            mutate_topology(plate)
            status = "MUTATION"

        # Logging
        f_mean = fracture.mean().item()
        log_fracture.append(f_mean)

        if t % 50 == 0:
            print(f"   Step {t:03d} | Fracture: {f_mean:.4f} | Status: {status}")

    # Save State (Scars persist to next run)
    torch.save({'plate': plate.state_dict(), 'grid': grid}, CHECKPOINT_FILE)
    print(f"   💾 Checkpoint saved. Organism preserved.")

    # Plotting
    plt.figure()
    plt.plot(log_fracture)
    plt.title(f"Fracture Rate - Run {mode}")
    plt.xlabel("Time")
    plt.ylabel("System Pain")
    plt.savefig(f"log_run_{mode}.png")
    print(f"   📊 Plot saved to log_run_{mode}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["A", "B", "C"], required=True)
    args = parser.parse_args()
    run_experiment(args.mode)