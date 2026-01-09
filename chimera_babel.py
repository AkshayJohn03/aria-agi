import torch
import torch.nn as nn
import copy
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
import os

# --- CONFIG ---
GRID_SIZE = 24          # Large enough for two islands
HIDDEN_DIM = 64
STEPS_PER_GEN = 24
MUTATION_RATE = 0.05
DECAY_RATE = 0.92       # <--- THE BABEL FACTOR (Lower = More Viscous/Isolated)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FILE = "chimera_babel_dna.pt"

print(f"🗼 TOWER OF BABEL PROTOCOL INITIATED ON {DEVICE}")
print(f"🌊 Physics Viscosity: {(1-DECAY_RATE)*100:.1f}% per step")

# --- 1. HARVEST SOULS (Same as Chimera) ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()

def get_vector(word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model(**inputs).last_hidden_state[0].mean(dim=0)

projector = torch.randn(HIDDEN_DIM, 768).to(DEVICE)

# Concept 1: Royalty
v_king = torch.matmul(projector, get_vector("king"))
v_man = torch.matmul(projector, get_vector("man"))
v_woman = torch.matmul(projector, get_vector("woman"))
target_royal = v_king - v_man + v_woman 

# Concept 2: Geography
v_paris = torch.matmul(projector, get_vector("paris"))
v_france = torch.matmul(projector, get_vector("france"))
v_germany = torch.matmul(projector, get_vector("germany"))
target_geo = v_paris - v_france + v_germany

# --- 2. THE VISCOUS ORGANISM ---
class CognitiveCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_perceive = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM * 3) * 0.1)
        self.w_react = nn.Parameter(torch.zeros(HIDDEN_DIM))

    def forward(self, grid):
        # Perception (Sobel)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(DEVICE) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        grad_x = torch.cat([torch.nn.functional.conv2d(grid[:, i:i+1], sobel_x, padding=1) for i in range(HIDDEN_DIM)], 1)
        grad_y = torch.cat([torch.nn.functional.conv2d(grid[:, i:i+1], sobel_y, padding=1) for i in range(HIDDEN_DIM)], 1)
        
        perception = torch.cat([grid, grad_x, grad_y], dim=1).permute(0, 2, 3, 1)
        update = torch.matmul(perception, self.w_perceive.t()) + self.w_react
        
        # Stochastic firing
        mask = (torch.rand_like(update[:, :, :, 0]) > 0.5).float().unsqueeze(-1)
        
        # --- THE BABEL UPDATE RULE ---
        # We multiply old grid by DECAY_RATE.
        # This means distant signals fade out before they cross the grid.
        new_grid = (grid.permute(0, 2, 3, 1) * DECAY_RATE) + (update * mask * 0.1)
        
        return torch.tanh(new_grid.permute(0, 3, 1, 2))

organism = CognitiveCell().to(DEVICE)

# Lazarus Load
if os.path.exists(SAVE_FILE):
    print("⚰️ RESURRECTING BABEL ORGANISM...")
    organism.load_state_dict(torch.load(SAVE_FILE))

# --- 3. EVOLUTION LOOP ---
best_fitness = -100.0

try:
    for generation in range(100000):
        # Mutate
        mutant = copy.deepcopy(organism)
        with torch.no_grad():
            mutant.w_perceive.add_(torch.randn_like(mutant.w_perceive) * MUTATION_RATE)
            mutant.w_react.add_(torch.randn_like(mutant.w_react) * MUTATION_RATE)

        # Run Physics
        live_grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
        # Inject Seeds at corners
        live_grid[:, :, 4, 4] = v_king.view(1, HIDDEN_DIM)
        live_grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_paris.view(1, HIDDEN_DIM)

        with torch.no_grad():
            for t in range(STEPS_PER_GEN):
                live_grid = mutant(live_grid)

        # Split Brain Judgment
        # Zone A (Royalty)
        zone_a = live_grid[:, :, 0:GRID_SIZE//2, 0:GRID_SIZE//2]
        act_a = zone_a.abs().sum(dim=1).squeeze()
        act_a[4, 4] = 0 
        if act_a.max() == 0: vec_a = torch.zeros(HIDDEN_DIM).to(DEVICE)
        else: 
            y, x = (act_a == act_a.max()).nonzero(as_tuple=False)[0]
            vec_a = zone_a[0, :, y, x]

        # Zone B (Geography)
        zone_b = live_grid[:, :, GRID_SIZE//2:, GRID_SIZE//2:]
        act_b = zone_b.abs().sum(dim=1).squeeze()
        if act_b.max() == 0: vec_b = torch.zeros(HIDDEN_DIM).to(DEVICE)
        else:
            y, x = (act_b == act_b.max()).nonzero(as_tuple=False)[0]
            vec_b = zone_b[0, :, y, x]

        fit_royal = torch.nn.functional.cosine_similarity(vec_a.unsqueeze(0), target_royal.unsqueeze(0)).item()
        fit_geo   = torch.nn.functional.cosine_similarity(vec_b.unsqueeze(0), target_geo.unsqueeze(0)).item()

        # Fitness = How well do you do BOTH?
        total_fitness = min(fit_royal, fit_geo)

        if total_fitness > best_fitness:
            best_fitness = total_fitness
            organism = mutant
            torch.save(organism.state_dict(), SAVE_FILE)
            msg = "✅ ISLANDS FORMING"
        else:
            msg = "❌ CHAOS"

        if generation % 50 == 0:
            print(f"Gen {generation}: Combined {best_fitness:.4f} (Royal: {fit_royal:.2f} | Geo: {fit_geo:.2f}) | {msg}")

except KeyboardInterrupt:
    print("🛑 PAUSED.")