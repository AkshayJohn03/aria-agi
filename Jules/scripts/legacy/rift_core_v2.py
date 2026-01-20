import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 48
CHANNELS = 32
STEPS = 500  # Longer run to see adaptation

print(f"🌋 PROJECT RIFT v2: STABILIZED TECTONICS ON {DEVICE}")

# --- 1. THE REFERENCE (The Tuning Forks) ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()
VOCAB_EMBEDS = model.wte.weight.detach()

# Readout Matrix (Random Init)
readout_matrix = torch.randn(CHANNELS, 768).to(DEVICE) * 0.1

# --- 2. THE TECTONIC SUBSTRATE ---
class TectonicPlate(nn.Module):
    def __init__(self):
        super().__init__()
        # Laplacian Kernel for Stress
        self.k_stress = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]]).float().to(DEVICE)
        self.k_stress = self.k_stress.repeat(CHANNELS, 1, 1, 1)

        # Elasticity (Memory)
        self.elasticity = nn.Parameter(torch.ones(1, CHANNELS, GRID_SIZE, GRID_SIZE).to(DEVICE) * 2.0)

    def forward(self, grid, input_force):
        # A. STRESS
        stress = F.conv2d(grid, self.k_stress, padding=1, groups=CHANNELS)

        # B. FRACTURE CHECK
        # If Stress > Elasticity, Snap.
        fracture_mask = (stress.abs() > self.elasticity).float()

        # C. ELASTICITY HARDENING (Scarring)
        # Fractured areas get harder (learn)
        with torch.no_grad():
            self.elasticity.data += fracture_mask * 0.1
            self.elasticity.data = torch.clamp(self.elasticity.data, 0.1, 10.0)

        # D. DYNAMICS UPDATE (Stabilized)
        # 1. Normalize Input Force (Prevent Nuke)
        input_force = torch.tanh(input_force)

        # 2. Update Physics
        # Grid + StressRelief + Input - Damping
        update = grid + (stress * -0.05) + input_force - (grid * 0.05)

        # 3. Apply Fracture (Energy Dissipation)
        # If fractured, reduce energy to 10% (Heat Loss), don't explode
        update = torch.where(fracture_mask > 0, update * 0.1, update)

        # 4. SAFETY CLAMP (The "Speed of Light" limit)
        update = torch.clamp(update, -5.0, 5.0)

        return update, fracture_mask

# --- 3. THE SEISMOGRAPH (EGO) ---
class Seismograph(nn.Module):
    def check_integrity(self, fracture_mask):
        total_fracture = fracture_mask.mean().item()
        status = "Stable"
        if total_fracture < 0.0001: status = "Stagnant"
        if total_fracture > 0.01: status = "Active"
        if total_fracture > 0.10: status = "Traumatized"
        return total_fracture, status

# --- EXECUTION ---
plate = TectonicPlate().to(DEVICE)
ego = Seismograph().to(DEVICE)
grid = torch.randn(1, CHANNELS, GRID_SIZE, GRID_SIZE).to(DEVICE) * 0.1
input_projector = torch.randn(768, CHANNELS).to(DEVICE) * 0.1 # Lower magnitude

print("\n🚀 LAUNCHING SIMULATION...")
stream = ["The", "world", "is", "full", "of", "pain", "and", "love"]
stream_vecs = [VOCAB_EMBEDS[tokenizer.encode(w)[0]] for w in stream]

for t in range(STEPS):
    # Input
    word_vec = stream_vecs[t % len(stream)]

    # Project Force
    force = torch.zeros_like(grid)
    force_vec = torch.matmul(word_vec, input_projector)
    cy, cx = GRID_SIZE//2, GRID_SIZE//2
    force[:, :, cy-2:cy+2, cx-2:cx+2] = force_vec.view(1, CHANNELS, 1, 1)

    # Physics
    grid, fractures = plate(grid, force)

    # Check
    violence, status = ego.check_integrity(fractures)

    # Readout
    if t % 50 == 0:
        grid_mean = grid.mean(dim=(2,3))
        readout_vec = torch.matmul(grid_mean, readout_matrix)

        # Find closest word
        scores = torch.matmul(VOCAB_EMBEDS, readout_vec.T).squeeze()
        best_id = torch.argmax(scores).item()
        output_word = tokenizer.decode([best_id]).strip()

        print(f"Step {t}: Input='{stream[t%len(stream)]}' | Fracture={violence:.4f} ({status}) | Resonance='{output_word}'")

        # Hebbian Learning (Stabilized)
        # Readout aligns with Grid activity
        # If output matches input, strengthen connection
        readout_matrix += torch.matmul(grid_mean.T, word_vec.view(1, 768)) * 0.0001

        # Normalize readout to prevent drift
        readout_matrix = F.normalize(readout_matrix, p=2, dim=1)

print("\n✅ RIFT v2 Complete. Stability achieved.")