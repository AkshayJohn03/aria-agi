import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 48
CHANNELS = 32
STEPS = 300

DAMPING = 0.04
STRESS_GAIN = 0.15
DIFFUSION_RATE = 0.20
YIELD_RATE = 0.03
MAX_ELASTICITY = 5.0
MIN_ELASTICITY = 0.1

print(f"\n🌋 PROJECT RIFT — SEMANTIC TECTONICS ({DEVICE})")
print("   Meaning is Stress. Memory is Deformation.\n")

# =========================
# LANGUAGE (INTERPRETATION ONLY)
# =========================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()
VOCAB_EMBEDS = model.wte.weight.detach()  # [50257, 768]

# Readout (OBSERVATION ONLY)
readout_matrix = torch.randn(CHANNELS, 768, device=DEVICE) * 0.01

# =========================
# PHYSICS KERNELS
# =========================
laplacian = torch.tensor(
    [[0, -1, 0],
     [-1, 4, -1],
     [0, -1, 0]],
    dtype=torch.float32,
    device=DEVICE
).view(1, 1, 3, 3).repeat(CHANNELS, 1, 1, 1)

diffusion_kernel = torch.ones((CHANNELS, 1, 3, 3), device=DEVICE) / 9.0

# =========================
# INITIAL STATE
# =========================
grid = torch.randn(1, CHANNELS, GRID_SIZE, GRID_SIZE, device=DEVICE) * 0.1
elasticity = torch.ones_like(grid) * 1.0  # STATE, NOT PARAMETER

# Input projection (external force only)
input_projector = torch.randn(768, CHANNELS, device=DEVICE) * 0.05

# =========================
# EGO (HOMEOSTASIS)
# =========================
def ego_check(fracture_energy):
    if fracture_energy < 0.001:
        return "Stagnant"
    elif fracture_energy > 0.06:
        return "Traumatized"
    else:
        return "Stable"

# =========================
# INPUT STREAM
# =========================
stream = ["the", "world", "is", "full", "of", "pain", "and", "love"]
stream_vecs = [VOCAB_EMBEDS[tokenizer.encode(w)[0]] for w in stream]

# =========================
# SIMULATION LOOP
# =========================
for t in range(STEPS):

    # ---- INPUT AS FORCE (distributed impact) ----
    word_vec = stream_vecs[t % len(stream)]
    force_vec = torch.matmul(word_vec, input_projector)

    force = torch.zeros_like(grid)

    # impact at random plate boundary (not center)
    y = np.random.randint(0, GRID_SIZE)
    x = np.random.randint(0, GRID_SIZE)
    force[:, :, y, x] = force_vec

    # ---- STRESS COMPUTATION ----
    stress = F.conv2d(grid, laplacian, padding=1, groups=CHANNELS)

    # ---- DIFFUSION ----
    diffusion = F.conv2d(grid, diffusion_kernel, padding=1, groups=CHANNELS)

    # ---- FRACTURE (PLASTIC YIELD, NOT BINARY) ----
    excess = stress.abs() - elasticity
    yield_mask = torch.clamp(excess, min=0.0)

    # Elastic hardening (memory)
    elasticity += YIELD_RATE * yield_mask
    elasticity.clamp_(MIN_ELASTICITY, MAX_ELASTICITY)

    # Energy release at fracture
    fracture_release = -torch.sign(stress) * yield_mask

    # ---- UPDATE ----
    grid = (
        grid
        - STRESS_GAIN * stress
        + DIFFUSION_RATE * diffusion
        + fracture_release
        + force
        - DAMPING * grid
    )

    # ---- EGO OBSERVATION ----
    fracture_energy = yield_mask.mean().item()
    status = ego_check(fracture_energy)

    # ---- READOUT (INTERPRETATION ONLY) ----
    if t % 30 == 0:
        grid_mean = grid.mean(dim=(2, 3))
        readout_vec = torch.matmul(grid_mean, readout_matrix)

        scores = torch.matmul(VOCAB_EMBEDS, readout_vec.T).squeeze()
        best_id = torch.argmax(scores).item()
        word = tokenizer.decode([best_id]).strip()

        print(
            f"Step {t:03d} | "
            f"Input='{stream[t % len(stream)]}' | "
            f"Fracture={fracture_energy:.4f} ({status}) | "
            f"Resonance='{word}'"
        )

        # Optional: slow drift of interpretation only
        readout_matrix += 0.0005 * torch.matmul(grid_mean.T, word_vec.view(1, 768))

print("\n✅ RIFT simulation complete.")
print("Memory formed via irreversible topological deformation.")
