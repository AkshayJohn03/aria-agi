import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32  # Smaller grid for faster learning loops
CH = 16
DT = 0.1
STEPS = 1000

print(f"🦾 PROJECT KINESIS: ACTIVE MOTOR BABBLING ON {DEVICE}")
print("   (Hypothesis: System learns to 'Brace' (Act) to minimize pain, driven by energy cost.)")

# =====================
# SUBSTRATE (Active Matter)
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

        # ACTUATOR MAP (The Muscle Strength)
        # Determines where the system *can* push back
        self.muscle_strength = nn.Parameter(torch.ones(1, CH, GRID, GRID, device=DEVICE) * 0.1)

        # POLICY MAP (The Brain)
        # Maps Input (Cue) -> Action (Muscle Contraction)
        # Simplest physical mapping: Correlation weight
        self.reflex_weight = nn.Parameter(torch.zeros(1, CH, GRID, GRID, device=DEVICE))

    def step(self, external_force, cue_signal, mode="learn"):
        # 1. GENERATE ACTION (The Twitch)
        # Action = Reflex (Cue * Weight) + Random Babbling

        reflex_action = cue_signal * self.reflex_weight

        if mode == "learn":
            # Babbling: Random twitches to explore solutions
            noise = torch.randn_like(self.u) * 0.5
            action = reflex_action + noise
        else:
            # Test: Only use learned reflexes
            action = reflex_action

        # Apply Muscle Constraints (Energy Cost limits magnitude)
        action = torch.tanh(action) * self.muscle_strength

        # 2. WAVE PHYSICS
        # Internal Stress
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -1.0 * self.u # Base stiffness 1.0

        # Acceleration = Physics + External Force + INTERNAL ACTION
        # This is the key: The system fights the external force with its action
        accel = diffusion + restoring + external_force + action - (0.1 * self.v)

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 3. MEASURE PAIN (Objective Function)
        # Pain = Physical Stress + Energy Cost of Action
        # Ideally, we want to minimize Stress, but Action is expensive.
        physical_stress = self.u.abs()
        energy_cost = action.abs() * 0.1 # Movement costs energy
        total_pain = physical_stress + energy_cost

        # 4. LEARNING (Reinforcement)
        # If Action reduced Pain compared to baseline -> Strengthen Weight
        # We approximate "Baseline" as the stress we would have felt without action (heuristic)

        with torch.no_grad():
            if mode == "learn":
                # Heuristic: If Pain is LOW but Action was HIGH, it means Action worked.
                # If Pain is HIGH and Action was HIGH, Action failed.

                # Inverted Hebbian:
                # Reward = (Action) * (1.0 - Pain)
                # If we acted and pain was low, Reinforce.
                # If we acted and pain was high, Punish.

                reward_signal = action * (0.5 - total_pain)

                # Update weights (Slowly)
                self.reflex_weight.data += reward_signal * cue_signal * 0.05
                self.reflex_weight.data.clamp_(-2.0, 2.0) # Can learn to push or pull

        return self.u, total_pain.mean().item(), action.abs().mean().item()

# =====================
# RUNNER
# =====================
def run():
    system = ActivePlate().to(DEVICE)

    cy, cx = GRID // 2, GRID // 2

    print("\n👶 PHASE 1: MOTOR BABBLING (Learning to Brace)")

    for t in range(STEPS):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)

        # CUE at t=10 (The "Warning")
        if t % 50 == 10:
            cue[:,:,cy-4:cy+4,cx-4:cx+4] = 1.0

        # IMPACT at t=15 (The "Punch")
        # Note: Very short gap. We need a fast reflex.
        if t % 50 == 15:
            force[:,:,cy-2:cy+2,cx-2:cx+2] = 5.0

        # Physics Step
        # System sees Cue, Feels Force, Tries to Act
        u, pain, effort = system.step(force, cue, mode="learn")

        if t % 100 == 0:
            print(f"   Step {t:03d} | Pain: {pain:.4f} | Effort (Action): {effort:.4f}")

    # RESET STATE
    system.u.zero_()
    system.v.zero_()

    print("\n🥋 PHASE 2: THE BRACING TEST (Action vs No Action)")

    # CONTROL: Cue provided, but Muscles paralyzed (Weights zeroed temporarily)
    print("   Running Control (Paralyzed)...")
    saved_weights = system.reflex_weight.clone()
    system.reflex_weight.data.zero_()

    pain_control = 0
    for t in range(50):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        if t == 10: cue[:,:,cy-4:cy+4,cx-4:cx+4] = 1.0
        if t == 15: force[:,:,cy-2:cy+2,cx-2:cx+2] = 5.0

        u, p, e = system.step(force, cue, mode="test")
        if t >= 15: pain_control += p

    # TEST: Cue provided, Muscles Active (Restored Weights)
    print("   Running Test (Active Brace)...")
    system.u.zero_()
    system.v.zero_()
    system.reflex_weight.data = saved_weights

    pain_active = 0
    for t in range(50):
        force = torch.zeros_like(system.u)
        cue = torch.zeros_like(system.u)
        if t == 10: cue[:,:,cy-4:cy+4,cx-4:cx+4] = 1.0
        if t == 15: force[:,:,cy-2:cy+2,cx-2:cx+2] = 5.0

        u, p, e = system.step(force, cue, mode="test")
        if t >= 15: pain_active += p

    print("\n📊 KINESIS REPORT")
    print(f"   Pain (Paralyzed): {pain_control:.4f}")
    print(f"   Pain (Active):    {pain_active:.4f}")

    delta = pain_control - pain_active
    pct = (delta / pain_control) * 100 if pain_control > 0 else 0

    print(f"   Pain Reduction: {delta:.4f} ({pct:.1f}%)")

    if pct > 5.0:
        print("   ✅ SUCCESS: The system learned to brace against the impact.")
    else:
        print("   ❌ FAILURE: No effective motor strategy learned.")

if __name__ == "__main__":
    run()