import torch
import torch.nn as nn
import numpy as np
import copy
from transformers import GPT2Model, GPT2Tokenizer
import matplotlib.pyplot as plt
from IPython.display import clear_output

# --- CONFIGURATION (The Petri Dish Settings) ---
GRID_SIZE = 16          # Small grid for Day 1 speed
HIDDEN_DIM = 64         # Complexity of cell "thought"
STEPS_PER_GEN = 16      # How long cells "talk" before we judge them
MUTATION_RATE = 0.02    # How reckless the mutations are
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🧬 GENESIS PROTOCOL INITIATED ON {DEVICE}")

# --- STEP 1: HARVEST THE SOUL (GPT-2 Geometry) ---
print("🔮 Extracting Relative Geometry from GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()

def get_vector(word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        # Get the last hidden state, average across tokens
        outputs = model(**inputs)
        return outputs.last_hidden_state[0].mean(dim=0)

# The "Royal" Geometry
v_king = get_vector("king")
v_man = get_vector("man")
v_woman = get_vector("woman")
v_queen = get_vector("queen")

# Project down to Cell Size (64-dim) for the experiment
# We use a fixed random projection to compress the 768-dim GPT vectors into our tiny cells
projector = torch.randn(HIDDEN_DIM, 768).to(DEVICE)
v_king = torch.matmul(projector, v_king)
v_man = torch.matmul(projector, v_man)
v_woman = torch.matmul(projector, v_woman)
v_queen = torch.matmul(projector, v_queen)

# The "Holy Grail" Vector (The target relative geometry)
# Target: King - Man + Woman = Queen
target_geometry_vector = v_king - v_man + v_woman
print("✅ Soul Harvested. Target Geometry Locked.")

# --- STEP 2: THE CELLULAR ORGANISM (NCA) ---
class CognitiveCell(nn.Module):
    def __init__(self):
        super().__init__()
        # The "Brain" of the cell: A simple matrix that processes neighbor signals
        # No layers, just one raw reaction matrix.
        self.w_perceive = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM * 3) * 0.1) # 3 inputs: Self, Neighbor_X, Neighbor_Y
        self.w_react = nn.Parameter(torch.zeros(HIDDEN_DIM)) # Bias/Metabolism

    def forward(self, grid):
        # grid shape: [Batch, Hidden, H, W]
        
        # 1. Perception (Sobel Filters to sense neighbors)
        # We cheat slightly by using conv2d to simulate "sensing neighbors" quickly
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(DEVICE) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        
        # Sense gradients (neighbor differences)
        grad_x = torch.cat([torch.nn.functional.conv2d(grid[:, i:i+1], sobel_x, padding=1) for i in range(HIDDEN_DIM)], 1)
        grad_y = torch.cat([torch.nn.functional.conv2d(grid[:, i:i+1], sobel_y, padding=1) for i in range(HIDDEN_DIM)], 1)
        
        # Concatenate: [Self State, Gradient X, Gradient Y]
        # Shape: [Batch, Hidden*3, H, W]
        perception = torch.cat([grid, grad_x, grad_y], dim=1)
        
        # 2. Liquid Reaction (The Brain)
        # We apply the weights manually (Linear projection per pixel)
        # Reshape for matmul: [Batch, H, W, Hidden*3]
        perception = perception.permute(0, 2, 3, 1) 
        
        # DNA execution: update = perception @ weights
        update = torch.matmul(perception, self.w_perceive.t()) + self.w_react
        
        # 3. Time Dynamics (Liquid Update)
        # Stochastic update mask (cells don't always fire)
        stochastic_mask = (torch.rand_like(update[:, :, :, 0]) > 0.5).float().unsqueeze(-1)
        
        # New State = Old State + Update (Liquid Integration)
        new_grid = grid.permute(0, 2, 3, 1) + (update * stochastic_mask * 0.1)
        
        return torch.tanh(new_grid.permute(0, 3, 1, 2)) # Normalize to -1..1

# Initialize the Organism
organism = CognitiveCell().to(DEVICE)
grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)

# --- STEP 3: THE EVOLUTION LOOP (No Backprop) ---
print("🦠 Injecting Seed...")
# Seed the center with "King" concept
center = GRID_SIZE // 2
grid[:, :, center, center] = v_king.view(1, HIDDEN_DIM)

best_fitness = -100.0
history = []

print("⚡ BEGINNING EVOLUTION (Ctrl+C to stop)")

try:
    for generation in range(10000):
        # A. MITOSIS (Create a mutated clone of the organism)
        mutant = copy.deepcopy(organism)
        with torch.no_grad():
            # Add random noise to weights (Genetic Mutation)
            mutant.w_perceive.add_(torch.randn_like(mutant.w_perceive) * MUTATION_RATE)
            mutant.w_react.add_(torch.randn_like(mutant.w_react) * MUTATION_RATE)

        # B. LIFE (Run the mutant for T steps)
        # Reset grid for fairness
        live_grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
        live_grid[:, :, center, center] = v_king.view(1, HIDDEN_DIM) # Re-inject seed
        
        with torch.no_grad():
            for t in range(STEPS_PER_GEN):
                live_grid = mutant(live_grid)

        # C. JUDGMENT (The "Winner-Takes-Signal" Check)
        # Instead of mean, we look at the most active non-center cell
        # This forces the organism to "move" the concept
        activation = live_grid.abs().sum(dim=1).squeeze() # Heatmap of activity
        # Ignore the center (the seed) to force growth
        activation[center, center] = 0 
        
        # Find the "Winner" cell (highest energy)
        if activation.max() == 0:
            winner_vector = torch.zeros(HIDDEN_DIM).to(DEVICE) # Dead organism
        else:
            y, x = (activation == activation.max()).nonzero(as_tuple=False)[0]
            winner_vector = live_grid[0, :, y, x]

        # D. FITNESS (Cosine Similarity to Target Geometry)
        similarity = torch.nn.functional.cosine_similarity(winner_vector.unsqueeze(0), target_geometry_vector.unsqueeze(0))
        fitness = similarity.item()

        # E. SELECTION
        if fitness > best_fitness:
            best_fitness = fitness
            organism = mutant # The mutant becomes the new parent
            msg = "✅ IMPROVEMENT"
        else:
            msg = "❌ EXTINCTION"

        history.append(best_fitness)

        # F. VISUALIZATION (Every 50 gens)
        if generation % 50 == 0:
            print(f"Gen {generation}: Fitness {best_fitness:.4f} | {msg}")
            
except KeyboardInterrupt:
    print("🛑 Evolution Paused.")