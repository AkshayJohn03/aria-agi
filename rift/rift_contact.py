import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# CONFIG
# =====================
GRID = 32
CH = 8
DT = 0.05
STEPS_TRAIN = 1000
STEPS_TEST = 200

print(f"🚪 PROJECT CONTACT: INTERACTION GATING ON {DEVICE}")
print("   (Hypothesis: System learns to CLOSE the gate to prevent predicted trauma.)")

# =====================
# SUBSTRATE
# =====================
class ContactPlate(nn.Module):
    def __init__(self):
        super().__init__()

        lap = torch.tensor([[[[0.5, 1.0, 0.5],
                              [1.0, -6.0, 1.0],
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        # State
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)

        # Base stiffness
        self.k = 1.0

        # 🔑 CONTACT GATE (Learned Boundary)
        self.gate_weight = nn.Parameter(torch.randn(1, CH, GRID, GRID, device=DEVICE) * 0.01)

    def step(self, external_force, cue, mode="learn"):
        # =====================
        # CONTACT GATING
        # =====================
        raw_gate = cue * self.gate_weight
        gate = torch.sigmoid(raw_gate)  # ∈ (0,1)

        # External force enters through gate
        gated_force = external_force * gate

        # =====================
        # PHYSICS
        # =====================
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -self.k * self.u
        accel = diffusion + restoring + gated_force - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # =====================
        # PAIN & COST
        # =====================
        deformation = self.u.abs()
        gate_cost = (1.0 - gate) * 0.05  # closing gate is metabolically expensive
        pain = deformation + gate_cost

        # =====================
        # LEARNING
        # =====================
        if mode == "learn":
            with torch.no_grad():
                # Reinforce gate closure if pain is reduced
                reward = -pain
                update = cue * reward * 0.1
                self.gate_weight.data += update
                self.gate_weight.data.clamp_(-5.0, 5.0)

        return pain.mean().item(), gate.mean().item()

# =====================
# RUNNER
# =====================
def run():
    system = ContactPlate().to(DEVICE)

    cy, cx = GRID // 2, GRID // 2

    print("\n👶 PHASE 1: CONDITIONING (Learning When to Close)")
    for t in range(STEPS_TRAIN):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)

        if t % 50 == 10:
            cue[:, :, cy-4:cy+4, cx-4:cx+4] = 1.0

        if t % 50 == 20:
            force[:, :, cy-2:cy+2, cx-2:cx+2] = 5.0

        pain, gate = system.step(force, cue, mode="learn")

        if t % 100 == 0:
            print(f"   Step {t:03d} | Pain: {pain:.4f} | Gate(Open=1): {gate:.4f}")

    # =====================
    # TEST
    # =====================
    system.u.zero_()
    system.v.zero_()

    print("\n🧪 PHASE 2: TEST (Cued vs Uncued)")

    # CONTROL
    print("   Running Control (Gate Open)...")
    saved_weights = system.gate_weight.clone()
    system.gate_weight.data.zero_()

    pain_control = 0
    for t in range(40):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        if t == 10: cue[:, :, cy-4:cy+4, cx-4:cx+4] = 1.0
        if t == 20: force[:, :, cy-2:cy+2, cx-2:cx+2] = 5.0
        p, g = system.step(force, cue, mode="test")
        pain_control += p

    # TEST
    print("   Running Test (Learned Gate)...")
    system.u.zero_()
    system.v.zero_()
    system.gate_weight.data = saved_weights

    pain_test = 0
    for t in range(40):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        if t == 10: cue[:, :, cy-4:cy+4, cx-4:cx+4] = 1.0
        if t == 20: force[:, :, cy-2:cy+2, cx-2:cx+2] = 5.0
        p, g = system.step(force, cue, mode="test")
        pain_test += p

    print("\n📊 CONTACT REPORT")
    print(f"   Pain (Gate Open): {pain_control:.4f}")
    print(f"   Pain (Gate Learned): {pain_test:.4f}")

    delta = pain_control - pain_test
    pct = (delta / pain_control) * 100 if pain_control > 0 else 0

    print(f"   Pain Reduction: {delta:.4f} ({pct:.1f}%)")

    if pct > 10.0:
        print("   ✅ SUCCESS: System learned anticipatory interaction gating.")
    else:
        print("   ❌ FAILURE: Gate control did not emerge.")

if __name__ == "__main__":
    run()
