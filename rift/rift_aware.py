import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32
CH = 8
DT = 0.05
STEPS_TRAIN = 1000
STEPS_TEST = 200

print(f"👁️ PROJECT AWARE: TEMPORAL ELIGIBILITY TRACES ON {DEVICE}")
print("   (Hypothesis: A decaying trace field bridges the time gap between Cue and Impact.)")

# =====================
# SUBSTRATE
# =====================
class AwarePlate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0.5, 1.0, 0.5],
                              [1.0, -6.0, 1.0],
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        # PHYSICS STATE
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)

        # MEMORY FIELDS
        # 1. Long-Term Structure (The Bone)
        self.k_structure = nn.Parameter(torch.ones(1, CH, GRID, GRID, device=DEVICE) * 0.5)

        # 2. Short-Term Trace (The Ghost/STM) - NOT a parameter, just state
        self.trace = torch.zeros(1, CH, GRID, GRID, device=DEVICE)

        # 3. Association Weight (Linking Trace -> Stiffness)
        self.reflex_weight = nn.Parameter(torch.zeros(1, CH, GRID, GRID, device=DEVICE))

    def step(self, external_force, cue, mode="learn"):
        # 1. UPDATE TRACE (Short-Term Memory)
        # Trace spikes with Cue, then decays.
        # This bridges the gap: Cue at T10 leaves a trace at T20.
        with torch.no_grad():
            self.trace = self.trace * 0.95 + cue * 1.0  # Decay + Input

        # 2. PREPAREDNESS (Action)
        # If we have a Trace and a learned Reflex, we Stiffen.
        # Action = Trace * Weight
        # Note: We use Trace, not Cue. The Trace is available even after Cue vanishes.
        preparedness = self.trace * self.reflex_weight
        
        # Sigmoid to bound the stiffening (0 to +2.0 stiffness)
        brace = torch.sigmoid(preparedness) * 2.0

        # 3. PHYSICS
        # Effective K = Structure + Brace
        K_total = self.k_structure + brace

        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -K_total * self.u
        accel = diffusion + restoring + external_force - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 4. PAIN
        # Pain = Physical Deformation
        pain = self.u.abs()
        total_pain = pain.mean().item()

        # 5. LEARNING (Temporal Credit Assignment)
        with torch.no_grad():
            if mode == "learn":
                # Crucial Step: We use TRACE, not CUE, for learning.
                # If Pain occurs while Trace is still active, we learned something.
                
                # Logic: We want to MINIMIZE Pain.
                # If Trace is High and Pain is High -> We failed to brace enough? 
                # Or rather: We want to associate Trace with the need to Stiffen.
                
                # Simple Hebbian: If Trace is active and we have High Stress, 
                # we should have braced.
                # Signal = Trace * Pain
                learning_signal = self.trace * pain
                
                # Update weights to increase bracing next time
                self.reflex_weight.data += learning_signal * 0.1
                self.reflex_weight.data.clamp_(-1.0, 5.0)

        return total_pain, brace.mean().item()

# =====================
# RUNNER
# =====================
def run():
    system = AwarePlate().to(DEVICE)

    cy, cx = GRID // 2, GRID // 2

    print("\n👶 PHASE 1: CONDITIONING (Bridging the Time Gap)")
    
    for t in range(STEPS_TRAIN):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)

        # CUE at t=10 (Disappears at t=11)
        if t % 60 == 10:
            cue[:,:,cy-4:cy+4,cx-4:cx+4] = 1.0

        # IMPACT at t=25 (15 steps later!)
        # Without Trace, the system would have forgotten the Cue.
        if t % 60 == 25:
            force[:,:,cy-2:cy+2,cx-2:cx+2] = 5.0

        pain, brace = system.step(force, cue, mode="learn")

        if t % 100 == 0:
            print(f"   Step {t:03d} | Pain: {pain:.4f} | Brace: {brace:.4f}")

    # =====================
    # TEST
    # =====================
    system.u.zero_()
    system.v.zero_()
    system.trace.zero_()

    print("\n🧪 PHASE 2: TEST (Cued vs Uncued)")

    # CONTROL: No Cue
    print("   Running Control (No Cue)...")
    pain_control = 0
    for t in range(60):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        # No Cue
        if t == 25: force[:,:,cy-2:cy+2,cx-2:cx+2] = 5.0
        
        p, b = system.step(force, cue, mode="test")
        pain_control += p

    # TEST: Cued
    print("   Running Test (With Cue)...")
    system.u.zero_()
    system.v.zero_()
    system.trace.zero_()
    
    pain_test = 0
    for t in range(60):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        
        # Cue at 10
        if t == 10: cue[:,:,cy-4:cy+4,cx-4:cx+4] = 1.0
        # Impact at 25
        if t == 25: force[:,:,cy-2:cy+2,cx-2:cx+2] = 5.0
        
        p, b = system.step(force, cue, mode="test")
        pain_test += p

    print("\n📊 AWARE REPORT")
    print(f"   Pain (Uncued): {pain_control:.4f}")
    print(f"   Pain (Cued):   {pain_test:.4f}")

    delta = pain_control - pain_test
    pct = (delta / pain_control) * 100 if pain_control > 0 else 0

    print(f"   Pain Reduction: {delta:.4f} ({pct:.1f}%)")

    if pct > 5.0:
        print("   ✅ SUCCESS: The 'Trace' allowed the system to predict across time.")
    else:
        print("   ❌ FAILURE: Temporal gap was too wide or bracing ineffective.")

if __name__ == "__main__":
    run()