import torch
import torch.nn as nn
import copy
from transformers import GPT2Model, GPT2Tokenizer
import numpy as np

# --- CONFIG ---
GRID_SIZE = 16
HIDDEN_DIM = 64
STEPS_PER_GEN = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🪓 TRAUMA TEST INITIATED ON {DEVICE}")

# --- 1. SETUP (Same as Genesis) ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()

def get_vector(word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model(**inputs).last_hidden_state[0].mean(dim=0)

projector = torch.randn(HIDDEN_DIM, 768).to(DEVICE) # Fixed projection
v_king = torch.matmul(projector, get_vector("king"))
v_man = torch.matmul(projector, get_vector("man"))
v_woman = torch.matmul(projector, get_vector("woman"))
v_queen = torch.matmul(projector, get_vector("queen"))
target = v_king - v_man + v_woman # The Holy Grail

# --- 2. THE ORGANISM ---
class CognitiveCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_perceive = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM * 3) * 0.1)
        self.w_react = nn.Parameter(torch.zeros(HIDDEN_DIM))

    def forward(self, grid):
        # ... (Same liquid logic as Genesis) ...
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(DEVICE) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        grad_x = torch.cat([torch.nn.functional.conv2d(grid[:, i:i+1], sobel_x, padding=1) for i in range(HIDDEN_DIM)], 1)
        grad_y = torch.cat([torch.nn.functional.conv2d(grid[:, i:i+1], sobel_y, padding=1) for i in range(HIDDEN_DIM)], 1)
        perception = torch.cat([grid, grad_x, grad_y], dim=1).permute(0, 2, 3, 1)
        update = torch.matmul(perception, self.w_perceive.t()) + self.w_react
        mask = (torch.rand_like(update[:, :, :, 0]) > 0.5).float().unsqueeze(-1)
        new_grid = grid.permute(0, 2, 3, 1) + (update * mask * 0.1)
        return torch.tanh(new_grid.permute(0, 3, 1, 2))

organism = CognitiveCell().to(DEVICE)

# --- 3. RAPID EVOLUTION (Get back to 0.7) ---
print("🚀 Re-evolving Organism to peak fitness...")
center = GRID_SIZE // 2
best_fitness = -1.0

for gen in range(2500): # Run until we hit the mark
    mutant = copy.deepcopy(organism)
    with torch.no_grad():
        mutant.w_perceive.add_(torch.randn_like(mutant.w_perceive) * 0.02)
        mutant.w_react.add_(torch.randn_like(mutant.w_react) * 0.02)
    
    grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
    grid[:, :, center, center] = v_king.view(1, HIDDEN_DIM)
    
    with torch.no_grad():
        for t in range(STEPS_PER_GEN):
            grid = mutant(grid)
            
    activation = grid.abs().sum(dim=1).squeeze()
    activation[center, center] = 0
    if activation.max() == 0: winner = torch.zeros(HIDDEN_DIM).to(DEVICE)
    else: 
        y, x = (activation == activation.max()).nonzero(as_tuple=False)[0]
        winner = grid[0, :, y, x]
        
    fitness = torch.nn.functional.cosine_similarity(winner.unsqueeze(0), target.unsqueeze(0)).item()
    
    if fitness > best_fitness:
        best_fitness = fitness
        organism = mutant
        if gen % 100 == 0: print(f"Gen {gen}: {best_fitness:.4f}")

    if best_fitness > 0.7:
        print(f"✅ PEAK FITNESS REACHED: {best_fitness:.4f}")
        break

# --- 4. THE LOBOTOMY (Trauma Test) ---
print("\n🔪 PERFORMING LOBOTOMY...")
print("Zeroing out 50% of the grid (Top-Left Quadrant & Random holes)...")

# Setup healthy grid
grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
grid[:, :, center, center] = v_king.view(1, HIDDEN_DIM)

# Run halfway (8 steps)
with torch.no_grad():
    for t in range(8):
        grid = organism(grid)

# APPLY TRAUMA (Delete 50% of data mid-thought)
mask = torch.ones_like(grid)
mask[:, :, 0:GRID_SIZE//2, 0:GRID_SIZE//2] = 0 # Nuke top-left corner
mask[:, :, :, :] *= (torch.rand_like(mask) > 0.2).float() # Random noise holes
grid = grid * mask 

print("🩸 Damage applied. Running remaining 8 steps WITHOUT repair...")

# Run remaining steps (injured)
with torch.no_grad():
    for t in range(8):
        grid = organism(grid) # The organism must route around the damage

# Measure Fitness
activation = grid.abs().sum(dim=1).squeeze()
activation[center, center] = 0
if activation.max() == 0: winner = torch.zeros(HIDDEN_DIM).to(DEVICE)
else: 
    y, x = (activation == activation.max()).nonzero(as_tuple=False)[0]
    winner = grid[0, :, y, x]
    
trauma_fitness = torch.nn.functional.cosine_similarity(winner.unsqueeze(0), target.unsqueeze(0)).item()

print(f"\n📊 RESULTS:")
print(f"Peak Health:  {best_fitness:.4f}")
print(f"After Trauma: {trauma_fitness:.4f}")

if trauma_fitness > 0.5:
    print("\n🏆 RESULT: ORGANISM SURVIVED. IT IS REGENERATIVE.")
    print("The system routed the logic around the dead cells.")
else:
    print("\n💀 RESULT: ORGANISM DIED. IT IS FRAGILE.")
    print("The system relies on fixed topology.")