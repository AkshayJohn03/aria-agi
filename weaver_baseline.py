import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import GPT2Model, GPT2Tokenizer

# --- CONFIG ---
GRID_SIZE = 24
HIDDEN_DIM = 64
DECAY_RATE = 0.92  # The Wall is active
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FILE = "chimera_babel_dna.pt" # Loading the Split-Brain Patient

print(f"🧶 PROJECT WEAVER: BASELINE PROTOCOL ON {DEVICE}")

# --- 1. HARVEST JOINT CONCEPTS ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()

def get_vector(word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model(**inputs).last_hidden_state[0].mean(dim=0)

projector = torch.randn(HIDDEN_DIM, 768).to(DEVICE) 

# The Inputs
v_king = torch.matmul(projector, get_vector("king"))
v_paris = torch.matmul(projector, get_vector("paris"))

# The Joint Target: "King + Paris" should loosely equal "France" or "Louis"
# Ideally: King - Man + Woman = Queen. 
# Here: King + Paris = ? (Let's use "France" + "Royal" as a proxy target vector)
# We define the target as the SUM of the two concepts. 
# If they merge, the grid should contain the sum vector.
target_joint = v_king + v_paris

# --- 2. THE ORGANISM (Standard Babel Physics) ---
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
        
        # Standard Viscosity (The Wall is UP)
        new_grid = (grid.permute(0, 2, 3, 1) * DECAY_RATE) + (update * mask * 0.1)
        return torch.tanh(new_grid.permute(0, 3, 1, 2))

organism = CognitiveCell().to(DEVICE)

if os.path.exists(SAVE_FILE):
    print(f"✅ LOADING PATIENT: {SAVE_FILE}")
    organism.load_state_dict(torch.load(SAVE_FILE))
else:
    print("❌ NO PATIENT FOUND. Run chimera_babel.py first.")
    exit()

# --- 3. THE JOINT TASK ---
print("\n🧩 TEST: JOINT REASONING (KING + PARIS)")

# Init Grid
grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)

# Inject BOTH seeds simultaneously
# King in Top-Left, Paris in Bottom-Right
grid[:, :, 4, 4] = v_king.view(1, HIDDEN_DIM)
grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_paris.view(1, HIDDEN_DIM)

# Run Thought Process
with torch.no_grad():
    for t in range(24):
        grid = organism(grid)

# --- 4. MEASURE BINDING ---
# We look for a "Binding Zone" - a place where the concepts merged.
# We check the CENTER of the grid.
center_zone = grid[:, :, GRID_SIZE//2-2:GRID_SIZE//2+2, GRID_SIZE//2-2:GRID_SIZE//2+2]
center_vector = center_zone.mean(dim=(2,3)).squeeze()

# Measure Fitness: Did the center contain the sum?
fitness = torch.nn.functional.cosine_similarity(center_vector.unsqueeze(0), target_joint.unsqueeze(0)).item()

print(f"   Center Binding Fitness: {fitness:.4f}")

if fitness > 0.5:
    print("   ⚠️ UNEXPECTED SUCCESS: The wall failed?")
else:
    print("   ✅ EXPECTED FAILURE: The concepts remained isolated.")
    print("      (This confirms we need a Thalamic Bridge)")