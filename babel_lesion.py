import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import GPT2Model, GPT2Tokenizer

# --- CONFIG ---
GRID_SIZE = 24
HIDDEN_DIM = 64
DECAY_RATE = 0.92  # The Viscosity that created the islands
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FILE = "chimera_babel_dna.pt"

print(f"🔪 SURGICAL PROTOCOL INITIATED ON {DEVICE}")

# --- 1. HARVEST SOULS ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()

def get_vector(word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model(**inputs).last_hidden_state[0].mean(dim=0)

projector = torch.randn(HIDDEN_DIM, 768).to(DEVICE) 

v_paris = torch.matmul(projector, get_vector("paris"))
v_france = torch.matmul(projector, get_vector("france"))
v_germany = torch.matmul(projector, get_vector("germany"))
target_geo = v_paris - v_france + v_germany

# --- 2. THE ORGANISM (Babel Physics) ---
class CognitiveCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_perceive = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM * 3) * 0.1)
        self.w_react = nn.Parameter(torch.zeros(HIDDEN_DIM))

    def forward(self, grid):
        # Perception
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(DEVICE) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        
        # FIXED: Removed the broken "groups=" line. 
        # We strictly iterate per channel to match the training logic.
        grad_x = torch.cat([F.conv2d(grid[:, i:i+1], sobel_x, padding=1) for i in range(HIDDEN_DIM)], 1)
        grad_y = torch.cat([F.conv2d(grid[:, i:i+1], sobel_y, padding=1) for i in range(HIDDEN_DIM)], 1)
        
        perception = torch.cat([grid, grad_x, grad_y], dim=1).permute(0, 2, 3, 1)
        update = torch.matmul(perception, self.w_perceive.t()) + self.w_react
        mask = (torch.rand_like(update[:, :, :, 0]) > 0.5).float().unsqueeze(-1)
        
        # Babel Viscosity Update
        new_grid = (grid.permute(0, 2, 3, 1) * DECAY_RATE) + (update * mask * 0.1)
        return torch.tanh(new_grid.permute(0, 3, 1, 2))

organism = CognitiveCell().to(DEVICE)

if os.path.exists(SAVE_FILE):
    print(f"✅ LOADING PATIENT: {SAVE_FILE}")
    organism.load_state_dict(torch.load(SAVE_FILE))
else:
    print("❌ NO PATIENT FOUND. Run chimera_babel.py first.")
    exit()

# --- 3. THE SURGERY ---

def run_simulation(mask=None):
    # Init Grid with Seed
    grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
    # Inject Geo Seed (Bottom Right)
    grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_paris.view(1, HIDDEN_DIM)
    
    # Run
    with torch.no_grad():
        for t in range(24):
            grid = organism(grid)
            if mask is not None:
                grid = grid * mask # Apply surgical clamp
    
    # Measure Geo Fitness
    zone_b = grid[:, :, GRID_SIZE//2:, GRID_SIZE//2:]
    act_b = zone_b.abs().sum(dim=1).squeeze()
    if act_b.max() == 0: vec_b = torch.zeros(HIDDEN_DIM).to(DEVICE)
    else:
        y, x = (act_b == act_b.max()).nonzero(as_tuple=False)[0]
        vec_b = zone_b[0, :, y, x]
        
    fitness = torch.nn.functional.cosine_similarity(vec_b.unsqueeze(0), target_geo.unsqueeze(0)).item()
    return fitness

# CONTROL GROUP (No Surgery)
print("\n🔎 RUNNING BASELINE (INTACT BRAIN)...")
baseline_fit = run_simulation(mask=None)
print(f"   Baseline Geography Fitness: {baseline_fit:.4f}")

# EXPERIMENTAL GROUP (Lobotomy)
print("\n🔪 PERFORMING LOBOTOMY (REMOVING TOP-LEFT QUADRANT)...")
surgical_mask = torch.ones(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
surgical_mask[:, :, 0:GRID_SIZE//2, 0:GRID_SIZE//2] = 0 # Kill Royal Zone

lesion_fit = run_simulation(mask=surgical_mask)
print(f"   Post-Surgery Geography Fitness: {lesion_fit:.4f}")

# --- DIAGNOSIS ---
drop = baseline_fit - lesion_fit
print("\n📋 DIAGNOSTIC REPORT:")
print(f"   Performance Drop: {drop:.4f}")

if abs(drop) < 0.05:
    print("   ✅ SUCCESS: INDEPENDENT MODULES CONFIRMED.")
    print("      The Geography Organ survived the death of the Royalty Organ.")
else:
    print("   ❌ FAILURE: ENTANGLEMENT DETECTED.")
    print("      The organs are still dependent.")