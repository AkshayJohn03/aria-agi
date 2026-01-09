import torch
import torch.nn as nn
import copy
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer

# --- CONFIG ---
GRID_SIZE = 16
HIDDEN_DIM = 64
STEPS_PER_GEN = 16
MUTATION_RATE = 0.05    # Increased for faster adaptation
DAMAGE_RATE = 0.20      # 20% of the brain dies every generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"⚔️ SPARTAN PROTOCOL INITIATED ON {DEVICE}")

# --- 1. SOUL HARVEST (Same as before) ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()

def get_vector(word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model(**inputs).last_hidden_state[0].mean(dim=0)

projector = torch.randn(HIDDEN_DIM, 768).to(DEVICE)
v_king = torch.matmul(projector, get_vector("king"))
v_man = torch.matmul(projector, get_vector("man"))
v_woman = torch.matmul(projector, get_vector("woman"))
v_queen = torch.matmul(projector, get_vector("queen"))
target = v_king - v_man + v_woman 

# --- 2. THE ORGANISM ---
class CognitiveCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_perceive = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM * 3) * 0.1)
        self.w_react = nn.Parameter(torch.zeros(HIDDEN_DIM))

    def forward(self, grid):
        # ... (Standard Liquid Logic) ...
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

# --- 3. THE SPARTAN EVOLUTION LOOP ---
center = GRID_SIZE // 2
best_fitness = -100.0

print("🛡️ EVOLUTION WITH DAMAGE ENABLED (Ctrl+C to stop)")

try:
    for generation in range(5000): # Longer run needed for robust evolution
        
        # A. MUTATION
        mutant = copy.deepcopy(organism)
        with torch.no_grad():
            mutant.w_perceive.add_(torch.randn_like(mutant.w_perceive) * MUTATION_RATE)
            mutant.w_react.add_(torch.randn_like(mutant.w_react) * MUTATION_RATE)

        # B. THE SPARTAN GAUNTLET (Run with Damage)
        live_grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
        live_grid[:, :, center, center] = v_king.view(1, HIDDEN_DIM)
        
        # Generate a random damage mask for THIS generation
        # 1.0 = Alive, 0.0 = Dead
        damage_mask = (torch.rand(1, 1, GRID_SIZE, GRID_SIZE).to(DEVICE) > DAMAGE_RATE).float()

        with torch.no_grad():
            for t in range(STEPS_PER_GEN):
                live_grid = mutant(live_grid)
                # APPLY DAMAGE EVERY STEP (Forces routing around holes)
                live_grid = live_grid * damage_mask 

        # C. JUDGMENT
        activation = live_grid.abs().sum(dim=1).squeeze()
        activation[center, center] = 0 
        
        if activation.max() == 0: 
            winner_vector = torch.zeros(HIDDEN_DIM).to(DEVICE)
        else:
            y, x = (activation == activation.max()).nonzero(as_tuple=False)[0]
            winner_vector = live_grid[0, :, y, x]

        similarity = torch.nn.functional.cosine_similarity(winner_vector.unsqueeze(0), target.unsqueeze(0))
        fitness = similarity.item()

        # D. SELECTION
        # We only accept the mutant if it performs better UNDER PRESSURE
        if fitness > best_fitness:
            best_fitness = fitness
            organism = mutant
            msg = "✅ ADAPTED"
        else:
            msg = "❌ DIED"

        if generation % 50 == 0:
            print(f"Gen {generation}: Fitness {best_fitness:.4f} (Damage: {DAMAGE_RATE*100}%) | {msg}")

except KeyboardInterrupt:
    print("🛑 Evolution Paused.")