import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 48
CHANNELS = 32
STEPS = 600

MIN_FRACTURE = 0.01
MAX_FRACTURE = 0.15

print(f"🔥 PROJECT RIFT v3 — ANTAGONIST EGO (EDGE OF CHAOS) [{DEVICE}]")

# ---------------- LANGUAGE REFERENCE (READ-ONLY) ----------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").to(DEVICE).eval()
VOCAB = model.wte.weight.detach()

readout_matrix = torch.randn(CHANNELS, 768, device=DEVICE) * 0.05

# ---------------- TECTONIC SUBSTRATE ----------------
class TectonicPlate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)
        self.k = lap.view(1,1,3,3).repeat(CHANNELS,1,1,1).to(DEVICE)

        self.elasticity = nn.Parameter(
            torch.ones(1, CHANNELS, GRID_SIZE, GRID_SIZE, device=DEVICE)
        )

    def forward(self, grid, external_force, ego_force):
        stress = F.conv2d(grid, self.k, padding=1, groups=CHANNELS)

        fracture = (stress.abs() > self.elasticity).float()

        # Memory: scar + decay
        with torch.no_grad():
            self.elasticity += fracture * 0.15
            self.elasticity *= 0.995
            self.elasticity.clamp_(0.4, 6.0)

        update = (
            grid
            - 0.08 * stress
            + torch.tanh(external_force)
            + ego_force
            - 0.03 * grid
        )

        # Fracture = forced reorganization, not damping
        update = torch.where(fracture > 0, -update, update)
        update.clamp_(-4.0, 4.0)

        return update, fracture

# ---------------- ANTAGONIST EGO ----------------
class AntagonistEGO:
    def regulate(self, grid, stress, fracture):
        level = fracture.mean().item()
        ego_force = torch.zeros_like(grid)
        status = "Optimal"

        # Too calm → amplify stress gradients
        if level < MIN_FRACTURE:
            status = "Bored → PRESSURIZING"
            ego_force = 0.6 * torch.sign(stress)

        # Too chaotic → oppose dominant gradients
        elif level > MAX_FRACTURE:
            status = "Overloaded → CONSTRAINING"
            ego_force = -0.4 * torch.tanh(stress)

        return ego_force, level, status

# ---------------- INIT ----------------
plate = TectonicPlate().to(DEVICE)
ego = AntagonistEGO()

grid = torch.randn(1, CHANNELS, GRID_SIZE, GRID_SIZE, device=DEVICE) * 0.1
input_proj = torch.randn(768, CHANNELS, device=DEVICE) * 0.08

stream = ["Love","Hate","Life","Death","Order","Chaos","Self","Void"]
stream_vecs = [VOCAB[tokenizer.encode(w)[0]] for w in stream]

print("\n🚀 SYSTEM AWAKENING…")

# ---------------- LOOP ----------------
for t in range(STEPS):
    word_vec = stream_vecs[t % len(stream)]
    force = torch.zeros_like(grid)

    fv = word_vec @ input_proj
    cy, cx = GRID_SIZE//2, GRID_SIZE//2
    force[:,:,cy-2:cy+2,cx-2:cx+2] = fv.view(1,CHANNELS,1,1)

    stress = F.conv2d(grid, plate.k, padding=1, groups=CHANNELS)
    fracture_pre = (stress.abs() > plate.elasticity).float()

    ego_force, frac_level, status = ego.regulate(grid, stress, fracture_pre)
    grid, fracture = plate(grid, force, ego_force)

    if t % 25 == 0:
        state = grid.mean(dim=(2,3))
        read = state @ readout_matrix
        token = torch.argmax(VOCAB @ read.T).item()
        word = tokenizer.decode([token]).strip()

        print(
            f"Step {t:03d} | "
            f"Fracture={frac_level:.4f} | "
            f"{status:<22} | "
            f"Resonance='{word}'"
        )

        # Gentle Hebbian drift
        readout_matrix += state.T @ word_vec.view(1,768) * 1e-4
        readout_matrix = F.normalize(readout_matrix, dim=1)

print("\n✅ RIFT v3 — Sustained Struggle Achieved")
