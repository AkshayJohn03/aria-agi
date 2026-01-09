import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import json

# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 48
CHANNELS = 32
STEPS = 500
CHECKPOINT = "rift_checkpoint.pt"

# =====================
# SUBSTRATE
# =====================
class TectonicPlate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0,-1,0],[-1,4,-1],[0,-1,0]]]], dtype=torch.float32)
        self.k_stress = lap.repeat(CHANNELS,1,1,1).to(DEVICE)
        self.elasticity = nn.Parameter(torch.ones(1,CHANNELS,GRID_SIZE,GRID_SIZE)*1.0)

    def forward(self, grid, force):
        stress = F.conv2d(grid, self.k_stress, padding=1, groups=CHANNELS)
        fracture = (stress.abs() > self.elasticity).float()

        with torch.no_grad():
            self.elasticity += fracture * 0.15
            self.elasticity *= 0.995
            self.elasticity.clamp_(0.5, 8.0)

        update = grid + stress * -0.1 + torch.tanh(force) - grid * 0.05
        update = torch.where(fracture > 0, update * 0.4, update)
        update = torch.clamp(update, -5, 5)

        return update, fracture

# =====================
# EGO
# =====================
class AntagonistEgo:
    def regulate(self, grid, fracture):
        f = fracture.mean().item()
        if f < 0.01:
            return torch.randn_like(grid) * 2.0, "PRESSURIZE", f
        if f > 0.15:
            return -grid * 0.6, "CONSTRAIN", f
        return torch.zeros_like(grid), "OPTIMAL", f

# =====================
# TOPOLOGY MUTATION (Run C)
# =====================
def topology_mutation(elasticity):
    mask = elasticity < 1.2
    elasticity[mask] *= 0.9
    elasticity[~mask] *= 1.05
    elasticity.clamp_(0.5, 10.0)

# =====================
# INPUT GENERATORS
# =====================
def simple_input(t):
    return torch.randn(1,CHANNELS,GRID_SIZE,GRID_SIZE)*0.1

def patterned_input(t):
    force = torch.zeros(1,CHANNELS,GRID_SIZE,GRID_SIZE)
    x = (t*3) % GRID_SIZE
    force[:,:,x:x+3,:] = 1.0
    return force

# =====================
# RUNNER
# =====================
def run(mode):
    plate = TectonicPlate().to(DEVICE)
    ego = AntagonistEgo()
    grid = torch.randn(1,CHANNELS,GRID_SIZE,GRID_SIZE).to(DEVICE)*0.1

    if os.path.exists(CHECKPOINT):
        data = torch.load(CHECKPOINT)
        plate.load_state_dict(data["plate"])
        grid = data["grid"]

    log = []

    for t in range(STEPS):
        if mode == "A":
            force = simple_input(t).to(DEVICE)
        else:
            force = patterned_input(t).to(DEVICE)

        ego_force, state, fracture = ego.regulate(grid, torch.zeros_like(grid))
        grid, frac = plate(grid, force + ego_force)

        if mode == "C" and t % 50 == 0:
            topology_mutation(plate.elasticity)

        log.append({
            "step": t,
            "fracture": frac.mean().item(),
            "ego": state
        })

    torch.save({
        "plate": plate.state_dict(),
        "grid": grid
    }, CHECKPOINT)

    with open(f"log_run_{mode}.json","w") as f:
        json.dump(log, f, indent=2)

    print(f"Run {mode} complete. Checkpoint saved.")

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["A","B","C"], required=True)
    args = parser.parse_args()
    run(args.mode)
