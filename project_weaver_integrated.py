import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import GPT2Model, GPT2Tokenizer

# --- CONFIG ---
GRID_SIZE = 24
HIDDEN_DIM = 64
MUTATION_RATE = 0.05
BASE_DECAY = 0.92       # The Wall (Isolation)
TOXIC_DECAY = 0.80      # The Death Zone (Binding Site default)
SAFE_DECAY = 0.99       # The Handshake (Binding Site active)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🧬 PROJECT WEAVER: INTEGRATED HANDSHAKE PROTOCOL ON {DEVICE}")

# --- 1. THE DICTIONARY (One Session to Rule Them All) ---
print("📚 Initializing Semantic Projector...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()
projector = torch.randn(HIDDEN_DIM, 768).to(DEVICE) # The Rosetta Stone

def get_vector(word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return torch.matmul(projector, model(**inputs).last_hidden_state[0].mean(dim=0))

# Harvest Vectors
v_king = get_vector("king")
v_paris = get_vector("paris")
v_france = get_vector("france")
v_germany = get_vector("germany")
# Targets
target_royal = get_vector("king") - get_vector("man") + get_vector("woman")
target_geo = get_vector("paris") - get_vector("france") + get_vector("germany")
target_joint = v_king + v_paris

# --- 2. THE ORGANISM ---
class CognitiveCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_perceive = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM * 3) * 0.1)
        self.w_react = nn.Parameter(torch.zeros(HIDDEN_DIM))

    def forward(self, grid, phase="babel"):
        # Perception
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(DEVICE) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        grad_x = torch.cat([F.conv2d(grid[:, i:i+1], sobel_x, padding=1) for i in range(HIDDEN_DIM)], 1)
        grad_y = torch.cat([F.conv2d(grid[:, i:i+1], sobel_y, padding=1) for i in range(HIDDEN_DIM)], 1)
        
        perception = torch.cat([grid, grad_x, grad_y], dim=1).permute(0, 2, 3, 1)
        update = torch.matmul(perception, self.w_perceive.t()) + self.w_react
        mask = (torch.rand_like(update[:, :, :, 0]) > 0.5).float().unsqueeze(-1)
        
        # --- PHYSICS ENGINE ---
        if phase == "babel":
            decay_map = torch.ones_like(grid) * BASE_DECAY
            
        elif phase == "weaver":
            decay_map = torch.ones_like(grid) * BASE_DECAY
            center_slice = (slice(None), slice(None), slice(GRID_SIZE//2-2, GRID_SIZE//2+2), slice(GRID_SIZE//2-2, GRID_SIZE//2+2))
            center_energy = grid[center_slice].abs().mean()
            
            center_decay = TOXIC_DECAY
            if center_energy > 0.15: 
                center_decay = SAFE_DECAY
            
            decay_map[center_slice] = center_decay

        # --- FIX APPLIED HERE ---
        # 1. Apply mask to update first (while both are H,W,C)
        masked_update = update * mask
        
        # 2. Permute result to match grid (C,H,W) and apply physics
        new_grid = (grid * decay_map) + (masked_update.permute(0, 3, 1, 2) * 0.1)
        
        return torch.tanh(new_grid)

organism = CognitiveCell().to(DEVICE)

# --- 3. PHASE 1: GROW THE BRAIN (Babel) ---
print("\n🏗️ PHASE 1: GROWING SPLIT BRAIN (Fast Track)...")
# We assume rapid convergence for brevity, simulating the Babel training
# In a real run, this loop restores the "Split Brain" state.
best_fitness = -1.0
optimizer = torch.optim.Adam(organism.parameters(), lr=0.01) # Using Adam for speed in this integrated test

for gen in range(200): # Shortened "Recap" Training
    optimizer.zero_grad()
    
    # Run Physics (Isolation)
    grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
    grid[:, :, 4, 4] = v_king.view(1, HIDDEN_DIM)
    grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_paris.view(1, HIDDEN_DIM)
    
    # We use mutation in real runs, but gradients here to quickly restore the state for the test
    # (Simulating a loaded model)
    # Actually, let's stick to Mutation to be pure BMOI, just fewer steps
    mutant = copy.deepcopy(organism)
    with torch.no_grad():
        mutant.w_perceive.add_(torch.randn_like(mutant.w_perceive) * MUTATION_RATE)
    
    with torch.no_grad():
        for t in range(16): grid = mutant(grid, phase="babel")
        
    # Measure
    vec_a = grid[:, :, 0:GRID_SIZE//2, 0:GRID_SIZE//2].mean(dim=(2,3)).squeeze()
    vec_b = grid[:, :, GRID_SIZE//2:, GRID_SIZE//2:].mean(dim=(2,3)).squeeze()
    
    fit_royal = F.cosine_similarity(vec_a.unsqueeze(0), target_royal.unsqueeze(0)).item()
    fit_geo = F.cosine_similarity(vec_b.unsqueeze(0), target_geo.unsqueeze(0)).item()
    
    if min(fit_royal, fit_geo) > best_fitness:
        best_fitness = min(fit_royal, fit_geo)
        organism = mutant
    
    if gen % 50 == 0:
        print(f"Gen {gen}: Royal {fit_royal:.2f} | Geo {fit_geo:.2f}")

print(f"🏆 BABEL READY. Best Fitness: {best_fitness:.2f}")

# --- 4. PHASE 2: THE HANDSHAKE TEST (Weaver) ---
print("\n🤝 PHASE 2: ACTIVATING BINDING SITE...")
print(f"{'STEP':<6} | {'CENTER E':<10} | {'BINDING':<10} | {'STATUS'}")
print("-" * 50)

# Init
grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
grid[:, :, 4, 4] = v_king.view(1, HIDDEN_DIM)
grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_paris.view(1, HIDDEN_DIM)

with torch.no_grad():
    for t in range(60):
        # Run with Weaver Physics
        grid = organism(grid, phase="weaver")
        
        # Measure Center
        center_zone = grid[:, :, GRID_SIZE//2-2:GRID_SIZE//2+2, GRID_SIZE//2-2:GRID_SIZE//2+2]
        center_energy = center_zone.abs().mean().item()
        center_vector = center_zone.mean(dim=(2,3)).squeeze()
        
        binding_fitness = F.cosine_similarity(center_vector.unsqueeze(0), target_joint.unsqueeze(0)).item()
        
        status = "☠️ Toxic (Decay)"
        if center_energy > 0.15:
            status = "✨ SAFE (Handshake!)"
            
        if t % 5 == 0:
            print(f"{t:<6} | {center_energy:.4f}     | {binding_fitness:.4f}     | {status}")