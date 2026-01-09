import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import GPT2Model, GPT2Tokenizer

# --- CONFIG ---
GRID_SIZE = 24
HIDDEN_DIM = 64
DECAY_RATE = 0.92
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FILE = "chimera_babel_dna.pt"

print(f"👁️ BABEL VISION DIAGNOSTIC ON {DEVICE}")

# --- 1. HARVEST TARGETS ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()
def get_vector(word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad(): return model(**inputs).last_hidden_state[0].mean(dim=0)

projector = torch.randn(HIDDEN_DIM, 768).to(DEVICE) # Note: In real loading, we'd need to save/load this random matrix too to be exact, 
# but for visualization of activity patterns, a fresh random proj is often okay if we just want to see heatmaps. 
# However, for accurate fitness checks, we ideally need the exact same projector. 
# *CRITICAL HACK:* For this diagnostic, we re-use the DNA but the fitness numbers might be off if we don't save the projector. 
# We care about SPATIAL PATTERNS more than exact numbers right now.

v_king = torch.matmul(projector, get_vector("king"))
v_paris = torch.matmul(projector, get_vector("paris"))
target_royal = torch.matmul(projector, get_vector("queen")) # Approx target for checking
target_geo = torch.matmul(projector, get_vector("berlin"))

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
        new_grid = (grid.permute(0, 2, 3, 1) * DECAY_RATE) + (update * mask * 0.1)
        return torch.tanh(new_grid.permute(0, 3, 1, 2))

organism = CognitiveCell().to(DEVICE)
if os.path.exists(SAVE_FILE):
    print(f"✅ Loaded DNA from {SAVE_FILE}")
    organism.load_state_dict(torch.load(SAVE_FILE))
else:
    print("❌ NO DNA FOUND. Run chimera_babel.py first!")
    exit()

# --- 3. RUN SIMULATION ---
grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
grid[:, :, 4, 4] = v_king.view(1, HIDDEN_DIM)
grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_paris.view(1, HIDDEN_DIM)

print("🧠 Thinking...")
with torch.no_grad():
    for t in range(24):
        grid = organism(grid)

# --- 4. VISUALIZE (fMRI) ---
# Sum absolute activity across hidden dim
activation = grid.abs().sum(dim=1).squeeze().cpu().numpy()

plt.figure(figsize=(6, 6))
plt.imshow(activation, cmap='inferno')
plt.title("Babel Organism: Dual Concept Activity")
plt.colorbar(label="Neural Activity")
# Mark seeds
plt.scatter([4], [4], c='cyan', label='King (Royal)')
plt.scatter([GRID_SIZE-5], [GRID_SIZE-5], c='lime', label='Paris (Geo)')
plt.legend()
plt.savefig("babel_brain_scan.png")
print("📸 Saved brain scan to babel_brain_scan.png")

# --- 5. SURGICAL LESION TEST ---
print("\n🔪 SURGICAL LESION TEST")
# Reset grid
grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
grid[:, :, 4, 4] = v_king.view(1, HIDDEN_DIM)
grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_paris.view(1, HIDDEN_DIM)

# Run halfway
with torch.no_grad():
    for t in range(12): grid = organism(grid)

# NUKE ROYALTY ZONE (Top Left)
print("⚠️ DESTROYING TOP-LEFT (ROYALTY) SECTOR...")
mask = torch.ones_like(grid)
mask[:, :, 0:GRID_SIZE//2, 0:GRID_SIZE//2] = 0
grid = grid * mask

# Run rest
with torch.no_grad():
    for t in range(12): grid = organism(grid)

# Check Geo Activity
zone_b = grid[:, :, GRID_SIZE//2:, GRID_SIZE//2:]
act_b = zone_b.abs().sum().item()
print(f"Geo Zone Activity after Royal Death: {act_b:.2f}")

if act_b > 10.0: # Arbitrary threshold for "alive"
    print("✅ SUCCESS: Geography survived the death of Royalty.")
    print("   The islands are functionally independent.")
else:
    print("❌ FAILURE: Geography collapsed.")
    print("   The system is still entangled.")