import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32
CH = 8
DT = 0.05
STEPS_TRAIN = 800
STEPS_TEST = 200

print(f"🕵️ PROJECT INQUISITOR: MEMORY STRESS TESTS ON {DEVICE}")

# =====================
# SUBSTRATE (Same as AWARE)
# =====================
class AwarePlate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0.5, 1.0, 0.5],
                              [1.0, -6.0, 1.0],
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        
        self.k_structure = nn.Parameter(torch.ones(1, CH, GRID, GRID, device=DEVICE) * 0.5)
        self.trace = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.reflex_weight = nn.Parameter(torch.zeros(1, CH, GRID, GRID, device=DEVICE))

    def step(self, external_force, cue, mode="learn", lobotomy=False):
        # 1. TRACE DYNAMICS
        with torch.no_grad():
            self.trace = self.trace * 0.95 + cue * 1.0 
            if lobotomy:
                self.trace.zero_() # Surgical removal of memory

        # 2. PREPAREDNESS
        preparedness = self.trace * self.reflex_weight
        brace = torch.sigmoid(preparedness) * 2.0

        # 3. PHYSICS
        K_total = self.k_structure + brace
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -K_total * self.u
        accel = diffusion + restoring + external_force - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 4. PAIN
        pain = self.u.abs().mean().item()

        # 5. LEARNING
        with torch.no_grad():
            if mode == "learn":
                learning_signal = self.trace * self.u.abs()
                self.reflex_weight.data += learning_signal * 0.1
                self.reflex_weight.data.clamp_(-1.0, 5.0)

        return pain, brace.mean().item()

    def reset_state(self):
        self.u.zero_()
        self.v.zero_()
        self.trace.zero_()

# =====================
# TEST HARNESS
# =====================
def run_test(system, test_name, gap_delay=15, use_noise_cue=False, do_lobotomy=False):
    print(f"\n🧪 TEST: {test_name}")
    print(f"   (Gap: {gap_delay}, Noise: {use_noise_cue}, Lobotomy: {do_lobotomy})")
    
    system.reset_state()
    
    # 1. Baseline (Uncued) for this specific gap
    pain_uncued = 0
    for t in range(gap_delay + 40):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        if t == gap_delay + 10: force[:,:,16-2:16+2,16-2:16+2] = 5.0
        p, b = system.step(force, cue, mode="test")
        pain_uncued += p

    system.reset_state()

    # 2. Test Run (Cued)
    pain_cued = 0
    for t in range(gap_delay + 40):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        
        # CUE
        if t == 10: 
            if use_noise_cue:
                cue = torch.randn_like(system.u) * 1.0 # Random Noise
            else:
                cue[:,:,16-4:16+4,16-4:16+4] = 1.0 # Correct Cue
        
        # IMPACT
        if t == 10 + gap_delay:
            force[:,:,16-2:16+2,16-2:16+2] = 5.0
            
        p, b = system.step(force, cue, mode="test", lobotomy=do_lobotomy)
        pain_cued += p

    delta = pain_uncued - pain_cued
    pct = (delta / pain_uncued) * 100 if pain_uncued > 0 else 0
    
    print(f"   Pain (Uncued): {pain_uncued:.4f}")
    print(f"   Pain (Cued):   {pain_cued:.4f}")
    print(f"   Reduction:     {pct:.2f}%")
    
    return pct

# =====================
# MAIN ROUTINE
# =====================
def run_suite():
    system = AwarePlate().to(DEVICE)
    cy, cx = GRID // 2, GRID // 2

    # --- TRAIN (Standard 15-step gap) ---
    print("🔨 TRAINING (Standard 15-step Gap)...")
    for t in range(STEPS_TRAIN):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        if t % 60 == 10: cue[:,:,cy-4:cy+4,cx-4:cx+4] = 1.0
        if t % 60 == 25: force[:,:,cy-2:cy+2,cx-2:cx+2] = 5.0
        system.step(force, cue, mode="learn")

    # --- VALIDATION SUITE ---
    
    # 1. CONTROL: Standard Test (Should pass)
    res_control = run_test(system, "Control (15-step Gap)", gap_delay=15)
    
    # 2. STRETCH TEST: Can it hold memory longer?
    res_stretch = run_test(system, "Stretch (40-step Gap)", gap_delay=40)
    
    # 3. NOISE TEST: Does random noise trigger it?
    res_noise = run_test(system, "Selectivity (Random Noise)", gap_delay=15, use_noise_cue=True)
    
    # 4. LOBOTOMY TEST: Does removing trace kill prediction?
    res_lobotomy = run_test(system, "Causality (Lobotomy)", gap_delay=15, do_lobotomy=True)

    print("\n📊 INQUISITOR FINAL VERDICT")
    print(f"   Control:   {res_control:.2f}% (Target: >5%)")
    print(f"   Stretch:   {res_stretch:.2f}% (Target: < Control, > 0%)")
    print(f"   Noise:     {res_noise:.2f}% (Target: ~0%)")
    print(f"   Lobotomy:  {res_lobotomy:.2f}% (Target: ~0%)")

    if res_control > 5.0 and res_noise < 1.0 and res_lobotomy < 1.0:
        print("\n✅ VALIDATED: Memory is real, selective, and causal.")
    else:
        print("\n❌ FAILED: Results indicate artifact or overfitting.")

if __name__ == "__main__":
    run_suite()