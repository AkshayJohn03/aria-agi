import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer

# --- CONFIG ---
GRID_SIZE = 24
HIDDEN_DIM = 64
MUTATION_RATE = 0.05
BASE_DECAY = 0.92
SNIPER_THRESHOLD = 0.01 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🎯 PROJECT WEAVER: PHASED SNIPER PROTOCOL (FIXED) ON {DEVICE}")

# --- 1. HARVEST LOGIC VECTORS ---
print("📚 Initializing Semantic Projector...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()
projector = torch.nn.Linear(768, HIDDEN_DIM, bias=False).to(DEVICE)
nn.init.orthogonal_(projector.weight)

def get_vector(word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[0].mean(dim=0)
        return projector(emb)

v_summer = get_vector("summer")
v_winter = get_vector("winter")
v_hot = get_vector("hot")
v_beach = get_vector("beach")
v_sauna = get_vector("sauna")

# --- 2. THE ORGANISM ---
class CognitiveCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_perceive = nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM, bias=False)
        self.w_react = nn.Parameter(torch.zeros(HIDDEN_DIM))
        nn.init.orthogonal_(self.w_perceive.weight)

    def forward(self, grid, prev_grid=None, sniper_active=False):
        batch, ch, h, w = grid.shape
        
        # Perception
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(DEVICE) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        grad_x = F.conv2d(grid, sobel_x.repeat(ch, 1, 1, 1), padding=1, groups=ch)
        grad_y = F.conv2d(grid, sobel_y.repeat(ch, 1, 1, 1), padding=1, groups=ch)
        
        perception = torch.cat([grid, grad_x, grad_y], dim=1).permute(0, 2, 3, 1)
        update = self.w_perceive(perception) + self.w_react.view(1, 1, 1, -1)
        
        # Physics Update
        mask = (torch.rand(batch, h, w, 1).to(DEVICE) > 0.0).float() 
        masked_update = (update * mask).permute(0, 3, 1, 2)
        new_grid = (grid * BASE_DECAY) + (masked_update * 0.1)
        new_grid = torch.tanh(new_grid)
        
        # --- THE PHASED SNIPER ---
        delta = 0.0 # <--- FIX: Initialize variable here
        
        if prev_grid is not None:
            cy, cx = h // 2, w // 2
            center_slice = (slice(None), slice(None), slice(cy-2, cy+2), slice(cx-2, cx+2))
            
            curr_center = new_grid[center_slice]
            prev_center = prev_grid[center_slice]
            
            # We calculate delta only if sniper is active to save compute, 
            # OR we can calculate it always for logging. Let's calculate only if active.
            if sniper_active:
                delta = (curr_center - prev_center).abs().mean()
                
                # THE SHOT
                if delta < SNIPER_THRESHOLD:
                    # Stagnation Detected -> DELETE THOUGHT
                    mask_sniper = torch.ones_like(new_grid)
                    mask_sniper[center_slice] = 0.0
                    new_grid = new_grid * mask_sniper
                
        return new_grid, delta

organism = CognitiveCell().to(DEVICE)

# --- 3. TRAINING WITH COMMIT WINDOWS ---
print("\n🏫 TRAINING: EXPLORE -> COMPETE -> COMMIT...")
best_fitness = -1.0
mutant = copy.deepcopy(organism)

# Phase Config
THINK_STEPS = 10
COMPETE_STEPS = 10 # Sniper is ON
COMMIT_STEPS = 10

TOTAL_STEPS = THINK_STEPS + COMPETE_STEPS + COMMIT_STEPS

for gen in range(500):
    # Mutate
    with torch.no_grad():
        noise = torch.randn_like(mutant.w_perceive.weight) * MUTATION_RATE
        mutant.w_perceive.weight.add_(noise)
    
    # Setup Contexts
    grid_a = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
    grid_a[:, :, 4, 4] = v_summer.view(1, 64)
    grid_a[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_hot.view(1, 64)
    
    grid_b = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
    grid_b[:, :, 4, 4] = v_winter.view(1, 64)
    grid_b[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_hot.view(1, 64)
    
    prev_a = None
    prev_b = None
    
    # Run Physics Loop
    with torch.no_grad():
        for t in range(TOTAL_STEPS):
            # Sniper Logic
            sniper_on = False
            if t >= THINK_STEPS and t < (THINK_STEPS + COMPETE_STEPS):
                sniper_on = True
            
            grid_a, _ = mutant(grid_a, prev_a, sniper_active=sniper_on)
            grid_b, _ = mutant(grid_b, prev_b, sniper_active=sniper_on)
            prev_a = grid_a.clone()
            prev_b = grid_b.clone()
            
    # Measure Output (Final State)
    cy, cx = GRID_SIZE // 2, GRID_SIZE // 2
    out_a = grid_a[:, :, cy-2:cy+2, cx-2:cx+2].mean(dim=(2,3)).squeeze()
    out_b = grid_b[:, :, cy-2:cy+2, cx-2:cx+2].mean(dim=(2,3)).squeeze()
    
    fit_a = F.cosine_similarity(out_a.unsqueeze(0), v_beach.unsqueeze(0)).item()
    fit_b = F.cosine_similarity(out_b.unsqueeze(0), v_sauna.unsqueeze(0)).item()
    
    avg_fitness = (fit_a + fit_b) / 2
    
    if avg_fitness > best_fitness:
        best_fitness = avg_fitness
        organism = copy.deepcopy(mutant)
    else:
        mutant = copy.deepcopy(organism)
        
    if gen % 50 == 0:
        print(f"Gen {gen}: Summer->Beach: {fit_a:.2f} | Winter->Sauna: {fit_b:.2f} | Avg: {avg_fitness:.3f}")

print(f"🏆 PHASED LOGIC TRAINED. Best Fitness: {best_fitness:.3f}")

# --- 4. INFERENCE: CHECKING THE DECISION ---
print("\n🕵️ FINAL DECISION CHECK...")

# Test Summer
grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
grid[:, :, 4, 4] = v_summer.view(1, 64)
grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_hot.view(1, 64)
prev = None
for t in range(TOTAL_STEPS):
    sniper_on = (t >= THINK_STEPS and t < THINK_STEPS + COMPETE_STEPS)
    grid, d = organism(grid, prev, sniper_active=sniper_on)
    prev = grid.clone()

out = grid[:, :, cy-2:cy+2, cx-2:cx+2].mean(dim=(2,3)).squeeze()
s_beach = F.cosine_similarity(out.unsqueeze(0), v_beach.unsqueeze(0)).item()
s_sauna = F.cosine_similarity(out.unsqueeze(0), v_sauna.unsqueeze(0)).item()
print(f"Context: SUMMER -> Beach: {s_beach:.2f} | Sauna: {s_sauna:.2f}")

# Test Winter
grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
grid[:, :, 4, 4] = v_winter.view(1, 64)
grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_hot.view(1, 64)
prev = None
for t in range(TOTAL_STEPS):
    sniper_on = (t >= THINK_STEPS and t < THINK_STEPS + COMPETE_STEPS)
    grid, d = organism(grid, prev, sniper_active=sniper_on)
    prev = grid.clone()

out = grid[:, :, cy-2:cy+2, cx-2:cx+2].mean(dim=(2,3)).squeeze()
s_beach_w = F.cosine_similarity(out.unsqueeze(0), v_beach.unsqueeze(0)).item()
s_sauna_w = F.cosine_similarity(out.unsqueeze(0), v_sauna.unsqueeze(0)).item()
print(f"Context: WINTER -> Beach: {s_beach_w:.2f} | Sauna: {s_sauna_w:.2f}")

# VERDICT
gap_a = s_beach - s_sauna
gap_b = s_sauna_w - s_beach_w

if gap_a > 0.1 and gap_b > 0.1:
    print(f"\n✅ SUCCESS: DECISION MADE. (Gap: {gap_a:.2f}, {gap_b:.2f})")
else:
    print(f"\n⚠️ FAILURE: Indecisive. (Gap: {gap_a:.2f}, {gap_b:.2f})")