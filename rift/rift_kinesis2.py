import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32
CH = 16
DT = 0.05
STEPS = 1000

print(f"🦾 PROJECT KINESIS-v2: IMPEDANCE CONTROL ON {DEVICE}")
print("   (Hypothesis: System learns to temporarily STIFFEN (Brace) to reject impact energy.)")

# =====================
# SUBSTRATE (Variable Impedance)
# =====================
class ActivePlate(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0.5, 1.0, 0.5], 
                              [1.0, -6.0, 1.0], 
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        # STATE
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        
        # PARAMETERS
        # Base Structure (Bone) - Permanent
        self.k_structure = nn.Parameter(torch.ones(1, CH, GRID, GRID, device=DEVICE) * 0.5)
        
        # POLICY MAP (The Brain)
        # Cue -> Action (Stiffness Modulation)
        # Weights initialized to small random values to break symmetry
        self.reflex_weight = nn.Parameter(torch.randn(1, CH, GRID, GRID, device=DEVICE) * 0.01)

    def step(self, external_force, cue_signal, mode="learn"):
        # 1. COMPUTE ACTION (Stiffness Modulation)
        # Action is strictly positive (you can't have negative stiffness)
        # Sigmoid ensures we stay within physical limits (0 to 1 range of modulation)
        
        raw_action = cue_signal * self.reflex_weight
        
        if mode == "learn":
            # Exploration noise
            noise = torch.randn_like(raw_action) * 0.1
            raw_action = raw_action + noise
            
        # Muscle Contraction (Bracing)
        # Adds to base stiffness. Max bracing = +2.0 stiffness.
        brace = torch.sigmoid(raw_action) * 2.0 
        
        # 2. PHYSICS (Impedance Control)
        # Effective K = Structure + Brace
        K_total = self.k_structure + brace
        
        # Wave Equation
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        # Restoring force depends on Dynamic Stiffness
        restoring = -K_total * self.u 
        
        accel = diffusion + restoring + external_force - (0.1 * self.v)
        
        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT
        
        # 3. MEASURE PAIN
        # Pain = Deformation (Damage) + Energy Cost of Bracing
        deformation = self.u.abs()
        metabolic_cost = brace * 0.05 # Stiffening is expensive
        
        total_pain = deformation + metabolic_cost
        
        # 4. LEARNING (Reinforcement)
        with torch.no_grad():
            if mode == "learn":
                # We want to minimize Pain.
                # If Brace was High and Pain was Low -> GOOD -> Increase Weight
                # If Brace was High and Pain was High -> BAD -> Decrease Weight
                
                # Simple reward signal: Deviation from average pain
                # (Simplified for stability: Inverse Pain)
                reward = -total_pain
                
                # Hebbian-like update: Cue * Action * Reward
                # If Cue was present, and we Braced, and Pain was low -> Reinforce
                update = cue_signal * brace * reward * 0.1
                
                self.reflex_weight.data += update
                # Clamp weights to prevent explosion
                self.reflex_weight.data.clamp_(-1.0, 5.0)

        return self.u, total_pain.mean().item(), brace.mean().item()

# =====================
# RUNNER
# =====================
def run():
    system = ActivePlate().to(DEVICE)
    
    cy, cx = GRID // 2, GRID // 2
    
    print("\n👶 PHASE 1: MOTOR LEARNING (Bracing)")
    
    # Track performance
    pain_history = []
    
    for t in range(STEPS):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        
        # CUE at t=10
        if t % 50 == 10:
            cue[:,:,cy-4:cy+4,cx-4:cx+4] = 1.0
            
        # IMPACT at t=15 (Short warning)
        if t % 50 == 15:
            force[:,:,cy-2:cy+2,cx-2:cx+2] = 5.0
            
        u, pain, brace = system.step(force, cue, mode="learn")
        pain_history.append(pain)
        
        if t % 100 == 0:
            avg_pain = np.mean(pain_history[-100:])
            print(f"   Step {t:03d} | Avg Pain: {avg_pain:.4f} | Avg Brace: {brace:.4f}")

    # RESET
    system.u.zero_()
    system.v.zero_()
    
    print("\n🥋 PHASE 2: THE BRACING TEST")
    
    # CONTROL: No Bracing (Weights Zeroed)
    print("   Running Control (Relaxed)...")
    saved_weights = system.reflex_weight.clone()
    system.reflex_weight.data.zero_()
    
    damage_relaxed = 0
    for t in range(50):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        if t == 10: cue[:,:,cy-4:cy+4,cx-4:cx+4] = 1.0
        if t == 15: force[:,:,cy-2:cy+2,cx-2:cx+2] = 5.0
        
        u, p, b = system.step(force, cue, mode="test")
        # Measure purely physical deformation (ignore metabolic cost for fairness)
        damage_relaxed += u.abs().mean().item()
        
    # TEST: Active Bracing
    print("   Running Test (Active Brace)...")
    system.u.zero_()
    system.v.zero_()
    system.reflex_weight.data = saved_weights
    
    damage_braced = 0
    for t in range(50):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        if t == 10: cue[:,:,cy-4:cy+4,cx-4:cx+4] = 1.0
        if t == 15: force[:,:,cy-2:cy+2,cx-2:cx+2] = 5.0
        
        u, p, b = system.step(force, cue, mode="test")
        damage_braced += u.abs().mean().item()

    print("\n📊 KINESIS REPORT")
    print(f"   Damage (Relaxed): {damage_relaxed:.4f}")
    print(f"   Damage (Braced):  {damage_braced:.4f}")
    
    delta = damage_relaxed - damage_braced
    pct = (delta / damage_relaxed) * 100
    
    print(f"   Damage Reduction: {delta:.4f} ({pct:.1f}%)")
    
    if pct > 10.0:
        print("   ✅ SUCCESS: System learned to stiffen upon seeing the cue.")
    else:
        print("   ❌ FAILURE: Bracing was ineffective or not learned.")

if __name__ == "__main__":
    run()