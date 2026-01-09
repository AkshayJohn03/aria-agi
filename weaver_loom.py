import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import GPT2Model, GPT2Tokenizer

# --- CONFIG ---
GRID_SIZE = 24
HIDDEN_DIM = 64
BASE_DECAY = 0.92      # The Wall
MAX_DECAY = 0.98       # The Bridge
HEAT_RATE = 0.02
COOL_RATE = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_FILE = "chimera_babel_dna.pt"

print(f"🧶 PROJECT WEAVER: THE LOOM (SCAFFOLDING) ON {DEVICE}")

# --- 1. HARVEST VECTORS ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()

def get_vector(word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return model(**inputs).last_hidden_state[0].mean(dim=0)

projector = torch.randn(HIDDEN_DIM, 768).to(DEVICE) 

# Specific Concepts
v_king = torch.matmul(projector, get_vector("king"))
v_paris = torch.matmul(projector, get_vector("paris"))
target_joint = v_king + v_paris

# Generic Schemas (The Scaffolding)
# We create "Ghost Vectors" to act as magnets in the center
v_schema_agent = torch.matmul(projector, get_vector("person")) # Generic Agent
v_schema_place = torch.matmul(projector, get_vector("place"))  # Generic Location

# --- 2. THE ORGANISM ---
class CognitiveCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_perceive = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM * 3) * 0.1)
        self.w_react = nn.Parameter(torch.zeros(HIDDEN_DIM))

    def forward(self, grid, current_decay, scaffold_mask=None):
        # Perception
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(DEVICE) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        
        grad_x = torch.cat([F.conv2d(grid[:, i:i+1], sobel_x, padding=1) for i in range(HIDDEN_DIM)], 1)
        grad_y = torch.cat([F.conv2d(grid[:, i:i+1], sobel_y, padding=1) for i in range(HIDDEN_DIM)], 1)
        
        perception = torch.cat([grid, grad_x, grad_y], dim=1).permute(0, 2, 3, 1)
        update = torch.matmul(perception, self.w_perceive.t()) + self.w_react
        mask = (torch.rand_like(update[:, :, :, 0]) > 0.5).float().unsqueeze(-1)
        
        # PHYSICS: Viscosity Update
        new_grid = (grid.permute(0, 2, 3, 1) * current_decay) + (update * mask * 0.1)
        
        # SCAFFOLDING INJECTION (The Loom)
        # If the Thalamus provides a scaffold, we gently add it to the grid
        if scaffold_mask is not None:
            # We add the scaffold vector (weighted small, just as a guide)
            new_grid = new_grid + (scaffold_mask.permute(0, 2, 3, 1) * 0.05)
            
        return torch.tanh(new_grid.permute(0, 3, 1, 2))

organism = CognitiveCell().to(DEVICE)

if os.path.exists(SAVE_FILE):
    print(f"✅ LOADING PATIENT: {SAVE_FILE}")
    organism.load_state_dict(torch.load(SAVE_FILE))
else:
    print("❌ NO PATIENT FOUND.")
    exit()

# --- 3. THE THALAMIC LOOP ---
print("\n🧠 ACTIVATING WORKING MEMORY TEMPLATE...")

grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
# Inject Memory Seeds
grid[:, :, 4, 4] = v_king.view(1, HIDDEN_DIM)
grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_paris.view(1, HIDDEN_DIM)

temperature = 0.0
current_decay = BASE_DECAY

print(f"{'STEP':<6} | {'TEMP':<6} | {'DECAY':<6} | {'BINDING':<10} | {'STATUS'}")
print("-" * 60)

with torch.no_grad():
    for t in range(80):
        # 1. Calculate Decay based on Heat
        current_decay = BASE_DECAY + (MAX_DECAY - BASE_DECAY) * temperature
        
        # 2. Build the Loom (Scaffold)
        # Only appears if Temperature (Attention) is high (> 0.3)
        scaffold = None
        if temperature > 0.3:
            scaffold = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
            # Project "Ghost Agent" to Center-Left
            scaffold[:, :, GRID_SIZE//2, GRID_SIZE//2-2] = v_schema_agent
            # Project "Ghost Place" to Center-Right
            scaffold[:, :, GRID_SIZE//2, GRID_SIZE//2+2] = v_schema_place
        
        # 3. Step
        grid = organism(grid, current_decay, scaffold)
        
        # 4. Measure
        center_zone = grid[:, :, GRID_SIZE//2-2:GRID_SIZE//2+2, GRID_SIZE//2-2:GRID_SIZE//2+2]
        center_vector = center_zone.mean(dim=(2,3)).squeeze()
        fitness = torch.nn.functional.cosine_similarity(center_vector.unsqueeze(0), target_joint.unsqueeze(0)).item()
        
        # 5. Thalamic Control
        status = "..."
        if fitness < 0.5:
            temperature += HEAT_RATE
            status = "🔥 Building Loom"
        elif fitness > 0.7:
            temperature -= COOL_RATE
            status = "💡 Weaving!"
        
        temperature = max(0.0, min(1.0, temperature))
        
        if t % 5 == 0:
            print(f"{t:<6} | {temperature:.2f}   | {current_decay:.3f}  | {fitness:.4f}     | {status}")