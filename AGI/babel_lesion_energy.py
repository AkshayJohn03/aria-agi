import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import GPT2Model, GPT2Tokenizer

# --- CONFIG ---
GRID_SIZE = 24
HIDDEN_DIM = 64
DECAY_RATE = 0.92
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FILE = "chimera_babel_dna.pt"

print(f"⚡ METABOLIC LESION PROTOCOL INITIATED ON {DEVICE}")

# --- 1. SETUP (No Projector needed for Energy check) ---
# We just need to inject a strong seed to wake it up.
# Since we lost the projector, we will inject a generic high-energy seed 
# to mimic the "Input" and see if it sustains.

# --- 2. THE ORGANISM ---
class CognitiveCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_perceive = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM * 3) * 0.1)
        self.w_react = nn.Parameter(torch.zeros(HIDDEN_DIM))

    def forward(self, grid):
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(DEVICE) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        
        grad_x = torch.cat([F.conv2d(grid[:, i:i+1], sobel_x, padding=1) for i in range(HIDDEN_DIM)], 1)
        grad_y = torch.cat([F.conv2d(grid[:, i:i+1], sobel_y, padding=1) for i in range(HIDDEN_DIM)], 1)
        
        perception = torch.cat([grid, grad_x, grad_y], dim=1).permute(0, 2, 3, 1)
        update = torch.matmul(perception, self.w_perceive.t()) + self.w_react
        mask = (torch.rand_like(update[:, :, :, 0]) > 0.5).float().unsqueeze(-1)
        
        new_grid = (grid.permute(0, 2, 3, 1) * DECAY_RATE) + (update * mask * 0.1)
        return torch.tanh(new_grid.permute(0, 3, 1, 2))

organism = CognitiveCell().to(DEVICE)

if os.path.exists(SAVE_FILE):
    print(f"✅ LOADING PATIENT: {SAVE_FILE}")
    organism.load_state_dict(torch.load(SAVE_FILE))
else:
    print("❌ NO PATIENT FOUND.")
    exit()

# --- 3. THE ENERGY TEST ---

def run_energy_test(mask=None):
    # Init Grid
    grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
    
    # INJECT RAW ENERGY (Since we lost the semantic projector)
    # We simulate an "Input Spike" at the Geography Seed Location
    # We use a random vector with magnitude 1.0
    seed_vector = torch.randn(1, HIDDEN_DIM).to(DEVICE)
    seed_vector = seed_vector / seed_vector.norm() * 10.0 # High energy spike
    
    # Inject into Geography Corner (Bottom Right)
    grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = seed_vector

    # Also Inject Royalty Corner (Top Left) to simulate full load
    grid[:, :, 4, 4] = seed_vector 

    # Run
    with torch.no_grad():
        for t in range(24):
            grid = organism(grid)
            if mask is not None:
                grid = grid * mask # Apply Surgical Clamp
    
    # MEASURE METABOLIC ACTIVITY IN GEOGRAPHY ZONE
    zone_b = grid[:, :, GRID_SIZE//2:, GRID_SIZE//2:]
    
    # Sum of absolute activations (Total Energy)
    total_energy = zone_b.abs().sum().item()
    return total_energy

# CONTROL GROUP
print("\n🔎 RUNNING BASELINE (INTACT)...")
baseline_energy = run_energy_test(mask=None)
print(f"   Baseline Geo Energy: {baseline_energy:.2f}")

# EXPERIMENTAL GROUP
print("\n🔪 PERFORMING LOBOTOMY (KILLING ROYALTY ZONE)...")
surgical_mask = torch.ones(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
surgical_mask[:, :, 0:GRID_SIZE//2, 0:GRID_SIZE//2] = 0 # Kill Top-Left

lesion_energy = run_energy_test(mask=surgical_mask)
print(f"   Post-Surgery Geo Energy: {lesion_energy:.2f}")

# --- DIAGNOSIS ---
# If the wall works, the energy should barely drop.
# If they are connected, the energy should plummet.
percent_survival = (lesion_energy / (baseline_energy + 1e-8)) * 100
print("\n📋 DIAGNOSTIC REPORT:")
print(f"   Survival Rate: {percent_survival:.1f}%")

if percent_survival > 90.0:
    print("   ✅ SUCCESS: TRUE INDEPENDENCE.")
    print("      The Geography Organ sustained its own metabolism.")
else:
    print("   ❌ FAILURE: METABOLIC COLLAPSE.")
    print("      The Geography Organ relied on energy from the Royalty Zone.")