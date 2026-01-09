import torch
import torch.nn as nn
import copy
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from transformers import GPT2Model, GPT2Tokenizer

# --- CONFIG ---
GRID_SIZE = 16
HIDDEN_DIM = 64
STEPS_PER_GEN = 16
DAMAGE_RATE = 0.20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNAPSHOT_DIR = "organism_scans"

if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

print(f"👁️ MRI PROTOCOL INITIATED ON {DEVICE}")

# --- 1. SOUL HARVEST ---
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

# --- 3. EVOLUTION WITH VISION ---
center = GRID_SIZE // 2
best_fitness = -100.0
snapshots = []

print("🎥 RECORDING EVOLUTION...")

try:
    for generation in range(1, 1001): # Short run for visualization
        
        # Mutation
        mutant = copy.deepcopy(organism)
        with torch.no_grad():
            mutant.w_perceive.add_(torch.randn_like(mutant.w_perceive) * 0.05)
            mutant.w_react.add_(torch.randn_like(mutant.w_react) * 0.05)

        # Run
        live_grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
        live_grid[:, :, center, center] = v_king.view(1, HIDDEN_DIM)
        damage_mask = (torch.rand(1, 1, GRID_SIZE, GRID_SIZE).to(DEVICE) > DAMAGE_RATE).float()

        with torch.no_grad():
            for t in range(STEPS_PER_GEN):
                live_grid = mutant(live_grid)
                live_grid = live_grid * damage_mask 

        # Fitness
        activation = live_grid.abs().sum(dim=1).squeeze()
        activation[center, center] = 0 
        
        if activation.max() == 0: 
            winner_vector = torch.zeros(HIDDEN_DIM).to(DEVICE)
        else:
            y, x = (activation == activation.max()).nonzero(as_tuple=False)[0]
            winner_vector = live_grid[0, :, y, x]

        fitness = torch.nn.functional.cosine_similarity(winner_vector.unsqueeze(0), target.unsqueeze(0)).item()

        if fitness > best_fitness:
            best_fitness = fitness
            organism = mutant
            
            # 📸 SNAPSHOT THE BRAIN
            # We visualize the aggregate activity (Energy Map)
            heatmap = activation.cpu().numpy()
            
            # Normalize for visualization
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            plt.figure(figsize=(4, 4))
            plt.imshow(heatmap, cmap='magma', interpolation='nearest')
            plt.title(f"Gen {generation} | Fit: {fitness:.2f}")
            plt.axis('off')
            
            filename = f"{SNAPSHOT_DIR}/gen_{generation}.png"
            plt.savefig(filename)
            plt.close()
            snapshots.append(filename)
            
            print(f"📸 Captured Gen {generation}: {best_fitness:.4f}")

    # COMPILE GIF
    print("🎞️ Compiling Time-Lapse...")
    with imageio.get_writer(f"{SNAPSHOT_DIR}/evolution.gif", mode='I', duration=0.2) as writer:
        for filename in snapshots:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    print(f"✅ DONE. Open {SNAPSHOT_DIR}/evolution.gif to see your organism.")

except KeyboardInterrupt:
    print("🛑 Stopped.")