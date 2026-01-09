import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import GPT2Model, GPT2Tokenizer

# --- CONFIG ---
GRID_SIZE = 24
HIDDEN_DIM = 64
MUTATION_RATE = 0.05
BASE_DECAY = 0.92      # Isolation
TOXIC_DECAY = 0.88     # Subcritical (Thinking Zone)
SAFE_DECAY = 0.99      # Safe (Solution Zone)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🧠 PROJECT WEAVER: CONTEXTUAL LOGIC GATE (v2 FIXED) ON {DEVICE}")

# --- 1. HARVEST LOGIC VECTORS ---
print("📚 Initializing Semantic Projector...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()
# Orthogonal Init (Standard)
projector = torch.nn.Linear(768, HIDDEN_DIM, bias=False).to(DEVICE)
nn.init.orthogonal_(projector.weight)

def get_vector(word):
    inputs = tokenizer(word, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[0].mean(dim=0)
        return projector(emb)

v_summer = get_vector("summer") # Context A
v_winter = get_vector("winter") # Context B
v_hot = get_vector("hot")       # Input (Ambiguous)

v_beach = get_vector("beach")   # Target A
v_sauna = get_vector("sauna")   # Target B

# --- 2. THE ORGANISM (Golden Standard + Weaver Physics) ---
class CognitiveCell(nn.Module):
    def __init__(self):
        super().__init__()
        # Use Linear layer for perception to match bmoi_core standard
        self.w_perceive = nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM, bias=False)
        self.w_react = nn.Parameter(torch.zeros(HIDDEN_DIM))
        nn.init.orthogonal_(self.w_perceive.weight)

    def forward(self, grid, phase="babel", stress=0.0):
        batch, ch, h, w = grid.shape
        
        # Perception
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(DEVICE) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        grad_x = F.conv2d(grid, sobel_x.repeat(ch, 1, 1, 1), padding=1, groups=ch)
        grad_y = F.conv2d(grid, sobel_y.repeat(ch, 1, 1, 1), padding=1, groups=ch)
        
        # [Batch, Channels*3, H, W] -> Permute to [B, H, W, C*3]
        perception = torch.cat([grid, grad_x, grad_y], dim=1).permute(0, 2, 3, 1)
        
        # Reaction: [B, H, W, C]
        update = self.w_perceive(perception) + self.w_react.view(1, 1, 1, -1)
        
        # Mask
        rand_map = torch.rand(batch, h, w, 1).to(DEVICE)
        mask = (rand_map > 0.0).float() # Full activity for logic test
        
        # Physics
        current_base = BASE_DECAY - stress
        decay_map = torch.ones(batch, h, w, 1).to(DEVICE) * current_base
        
        if phase == "weaver":
            cy, cx = h // 2, w // 2
            # Check Energy in Center
            center_energy = grid[:, :, cy-2:cy+2, cx-2:cx+2].abs().mean()
            
            center_decay = TOXIC_DECAY # Subcritical default
            if center_energy > 0.15:
                center_decay = SAFE_DECAY
            
            # Apply decay to center [B, H, W, 1]
            decay_map[:, cy-2:cy+2, cx-2:cx+2, :] = center_decay

        # Apply Physics
        # Update is [B, H, W, C], Grid is [B, C, H, W]
        masked_update = (update * mask).permute(0, 3, 1, 2)
        decay_map = decay_map.permute(0, 3, 1, 2)
        
        new_grid = (grid * decay_map) + (masked_update * 0.1)
        return torch.tanh(new_grid)

organism = CognitiveCell().to(DEVICE)

# --- 3. TRAINING: THE DUAL TEACHER ---
print("\n🏫 TRAINING: DUAL-CONTEXT CURRICULUM...")
best_fitness = -1.0
mutant = copy.deepcopy(organism)

for gen in range(400):
    # Mutate
    with torch.no_grad():
        noise = torch.randn_like(mutant.w_perceive.weight) * MUTATION_RATE
        mutant.w_perceive.weight.add_(noise)
    
    # CASE 1: Summer + Hot -> Beach
    grid_a = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
    # FIX: Reshape vectors to [1, 64] to match grid slice [1, 64]
    grid_a[:, :, 4, 4] = v_summer.view(1, HIDDEN_DIM) 
    grid_a[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_hot.view(1, HIDDEN_DIM)
    
    # CASE 2: Winter + Hot -> Sauna
    grid_b = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
    grid_b[:, :, 4, 4] = v_winter.view(1, HIDDEN_DIM)
    grid_b[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_hot.view(1, HIDDEN_DIM)
    
    # Run Physics
    with torch.no_grad():
        for t in range(24):
            grid_a = mutant(grid_a, phase="weaver")
            grid_b = mutant(grid_b, phase="weaver")
            
    # Measure Center Output
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

print(f"🏆 LOGIC GATE TRAINED. Best Fitness: {best_fitness:.3f}")

# --- 4. THE INFERENCE TEST ---
print("\n🕵️ RUNNING INFERENCE TEST...")
print("Can the organism switch outputs based on context?")

# Test Summer
grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
grid[:, :, 4, 4] = v_summer.view(1, HIDDEN_DIM)
grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_hot.view(1, HIDDEN_DIM)
for t in range(32): grid = organism(grid, phase="weaver")
out = grid[:, :, cy-2:cy+2, cx-2:cx+2].mean(dim=(2,3)).squeeze()
score_beach = F.cosine_similarity(out.unsqueeze(0), v_beach.unsqueeze(0)).item()
score_sauna = F.cosine_similarity(out.unsqueeze(0), v_sauna.unsqueeze(0)).item()
print(f"Context: SUMMER + HOT -> Beach ({score_beach:.2f}) vs Sauna ({score_sauna:.2f})")

# Test Winter
grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)
grid[:, :, 4, 4] = v_winter.view(1, HIDDEN_DIM)
grid[:, :, GRID_SIZE-5, GRID_SIZE-5] = v_hot.view(1, HIDDEN_DIM)
for t in range(32): grid = organism(grid, phase="weaver")
out = grid[:, :, cy-2:cy+2, cx-2:cx+2].mean(dim=(2,3)).squeeze()
score_beach = F.cosine_similarity(out.unsqueeze(0), v_beach.unsqueeze(0)).item()
score_sauna = F.cosine_similarity(out.unsqueeze(0), v_sauna.unsqueeze(0)).item()
print(f"Context: WINTER + HOT -> Beach ({score_beach:.2f}) vs Sauna ({score_sauna:.2f})")