import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from bmoi_core import CognitiveCell, DEVICE  # Import from our new core

# --- CONFIG ---
GRID_SIZE = 120        # MASSIVE SCALE
HIDDEN_DIM = 64
STEPS = 100            # More steps for signal to propagate
DECAY_RATE = 0.94      # Viscosity tuned for larger distances
SEED_LOCATIONS = [
    (10, 10),          # Top-Left (Kingdom A)
    (110, 110),        # Bottom-Right (Kingdom B)
    (10, 110),         # Top-Right (Kingdom C)
    (60, 60)           # Center (Kingdom D)
]

print(f"🌍 LAUNCHING 120x120 PLANETARY SIMULATION ON {DEVICE}...")

# 1. Init Organism (Randomly initialized for visual pattern test)
# In a real run, load 'chimera_babel_dna.pt'
organism = CognitiveCell(HIDDEN_DIM).to(DEVICE)

# 2. Init Grid
grid = torch.zeros(1, HIDDEN_DIM, GRID_SIZE, GRID_SIZE).to(DEVICE)

# 3. Inject Seeds (Four distinct concepts)
# We use random vectors to represent 4 different ideas
seeds = [torch.randn(HIDDEN_DIM).to(DEVICE) for _ in range(4)]

for i, (r, c) in enumerate(SEED_LOCATIONS):
    grid[:, :, r, c] = seeds[i]

# 4. Run Simulation
print("⏳ Evolving Physics...")
history = []

with torch.no_grad():
    for t in range(STEPS):
        grid = organism(grid, decay_rate=DECAY_RATE)
        
        if t % 10 == 0:
            # Capture total activity for visualization
            activity = grid.abs().mean(dim=1).squeeze().cpu().numpy()
            history.append(activity)
            print(f"   Step {t}/{STEPS} complete.")

# 5. Visualization (High Res)
print("📸 Rendering Satellite Imagery...")
plt.figure(figsize=(10, 10))
plt.imshow(history[-1], cmap='inferno')
plt.title(f"BMOI Scale Test: 120x120 Grid (t={STEPS})")
plt.colorbar(label="Neural Activity")
plt.savefig("bmoi_scale_120.png", dpi=300)
print("✅ Saved to bmoi_scale_120.png")