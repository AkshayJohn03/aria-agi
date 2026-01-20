import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math  # Added for float operations

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32
CH = 8
DT = 0.05

STEPS_TRAIN = 3000
STEPS_PER_EPISODE = 120

print("🧬 PROJECT RIFT: HYPER-COMPENSATION (FIXED v2) ON", DEVICE)
print("   (Hypothesis: Survived stress increases future capacity.)")

# =====================
# BODY (NON-DIFFERENTIABLE PHYSICS)
# =====================
class RiftGrowthBody:
    def __init__(self):
        # Laplacians for diffusion (Fixed physics constants)
        self.laplacian = torch.tensor(
            [[[[0.5, 1.0, 0.5],
               [1.0, -6.0, 1.0],
               [0.5, 1.0, 0.5]]]],
            device=DEVICE
        ).repeat(CH, 1, 1, 1)

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, GRID),
            torch.linspace(-1, 1, GRID),
            indexing="ij"
        )
        self.x_grid = x.to(DEVICE)
        self.y_grid = y.to(DEVICE)

        # EVOLUTIONARY PARAMETERS
        self.health_max = 1.0
        self.battery_max = 1.0
        self.barrier_strength = 2.0

        self.reset()

    def reset(self):
        # Physical fields
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros_like(self.u)
        self.trace = torch.zeros_like(self.u)

        # Posture state
        self.posture = torch.tensor([0.0, 0.0], device=DEVICE)
        self.vel = torch.zeros_like(self.posture)

        # Physiological state
        self.health = self.health_max
        self.stability = 1.0
        self.battery = self.battery_max * 0.5

        # Markers for evolution
        self.used_impulse = False
        self.survived_stress = False

    # ---- PURE PHYSICS STEP (NO AUTOGRAD) ----
    # All inputs here are floats (scalars), not Tensors
    @torch.no_grad()
    def physics_step(self, force, cue, melt, tilt, fire):
        # 1. Update Trace
        self.trace = self.trace * 0.85 + cue

        # 2. Dissonance (Belief vs Reality)
        belief = self.trace.mean().item()
        reality = self.posture[0].item()

        dissonance = 0.0
        if abs(belief) > 0.1 and (belief * reality < 0):
            dissonance = 0.6
            self.stability -= dissonance * 0.04
            self.survived_stress = True

        self.stability = max(0.0, min(1.0, self.stability))

        # 3. Battery Dynamics
        self.battery = min(self.battery + 0.02, self.battery_max)

        impulse = 0.0
        # Fire logic: uses float math
        if fire > 0.5 and self.battery > 0.6 * self.battery_max:
            # FIX: Use math.copysign or simple conditional for floats
            sign_tilt = 1.0 if tilt > 0 else -1.0
            impulse = sign_tilt * self.battery * 6.0
            self.battery = 0.0
            self.used_impulse = True

        # 4. Landscape Forces
        barrier = self.barrier_strength * (1.0 - melt)
        px = self.posture[0].item()
        target = 0.6

        # Force = -dV/dx
        fx = -4.0 * barrier * px * (px**2 - target**2) + (tilt * 0.4) + impulse
        fy = -2.0 * self.posture[1].item()

        damping = 0.5 + (1.0 - self.stability) * 0.5

        # 5. Motion Integration
        # We temporarily use tensors for vector math, but detach them immediately conceptually
        self.vel += torch.tensor([fx, fy], device=DEVICE) * DT
        self.vel *= (1.0 - damping * DT)
        self.posture = torch.clamp(self.posture + self.vel * DT, -1.0, 1.0)

        # 6. Field Physics (Shields)
        dist = (self.x_grid - self.posture[0])**2 + (self.y_grid - self.posture[1])**2
        shield = torch.exp(-dist / (2 * 0.4**2))
        shield = shield.unsqueeze(0).unsqueeze(0).expand(1, CH, GRID, GRID)

        eff_force = force * (1.0 - shield)
        diff = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)

        accel = diff - self.u + eff_force - 0.1 * self.v
        self.v += accel * DT
        self.u += self.v * DT

        # 7. Damage
        damage = 0.0
        max_deform = self.u.abs().max().item()
        if max_deform > 0.1:
            damage = max_deform * 0.4
            self.health -= damage

        dead = self.health <= 0.0

        return dissonance, damage, dead

    def evolve(self):
        # HYPER-COMPENSATION LOGIC
        # If we used the battery (and survived), we need a bigger battery next time.
        if self.used_impulse:
            self.battery_max += 0.1

        # If we faced dissonance/stress (and survived), we harden our health.
        if self.survived_stress and self.health > 0:
            self.health_max += 0.1

        # If we died, the barrier was too weak/confusing?
        # Or maybe we just reset parameters to ensure we don't drift into weakness.
        if self.health <= 0:
            # Penalty for death: Barrier weakens slightly (requires more active control next time)
            # or just reset evolution slightly.
            self.barrier_strength = max(1.0, self.barrier_strength * 0.95)

    def state(self):
        # Return state as a detached Tensor for the Brain
        return torch.tensor([
            self.trace.mean().item(),
            self.posture[0].item(),
            self.battery / self.battery_max, # Normalize input for brain
            self.stability,
            max(0.0, self.health / self.health_max), # Normalize health
            self.battery_max # Context: Brain knows its capacity changes
        ], device=DEVICE)

# =====================
# BRAIN (DIFFERENTIABLE)
# =====================
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 3)

    def forward(self, x):
        o = self.fc(x)
        # Melt (Sigmoid), Tilt (Tanh), Fire (Sigmoid)
        return torch.sigmoid(o[0]), torch.tanh(o[1]), torch.sigmoid(o[2])

# =====================
# RUNNER
# =====================
def run():
    body = RiftGrowthBody()
    brain = Brain().to(DEVICE)
    opt = optim.Adam(brain.parameters(), lr=0.01)

    print("\n🧬 STARTING EVOLUTION...")

    for ep in range(STEPS_TRAIN):
        body.reset()

        # Create a tensor for loss accumulation
        total_pain = torch.tensor(0.0, device=DEVICE)

        impact = random.randint(80, 110)
        cue1 = impact - 60
        cue2 = impact - 20

        # Randomize Left/Right scenarios
        if random.random() > 0.5:
            c1, c2 = 1.0, -1.0
            impact_slice = (slice(14,18), slice(22,26)) # Right side
        else:
            c1, c2 = -1.0, 1.0
            impact_slice = (slice(14,18), slice(6,10)) # Left side

        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
            if t == cue1: cue[:] = c1
            if t == cue2: cue[:] = c2 * 2.0 # Stronger counter-cue

            force = torch.zeros_like(cue)
            if impact <= t < impact + 5:
                force[:, :, impact_slice[0], impact_slice[1]] = 20.0

            # 1. Get State (Detached)
            state = body.state()

            # 2. Brain Decision (Differentiable)
            melt, tilt, fire = brain(state)

            # 3. Physics Step (Non-Differentiable)
            # Pass floats to physics engine
            dissonance, damage, dead = body.physics_step(
                force, cue, melt.item(), tilt.item(), fire.item()
            )

            # 4. Reconstruct Loss (The Bridge)
            # We must penalize the ACTIONS based on the OUTCOMES.
            # Since 'dissonance' and 'damage' are floats, we multiply them by constant tensors
            # but we simply add them to the metabolic costs which ARE attached to the graph.

            # The Brain feels the pain via the metabolic cost connection
            # and the magnitude of the pain signal.

            # Pain = Real Pain (Float) + Regulatory Cost (Gradient)
            # We treat the float pain as a scaler for the metabolic tensor?
            # No, standard RL/Evolution strategy:
            # Here we use the "Direct Pain" method:
            # loss = (pain_value) * (log_prob) ? No, this is continuous control.

            # Simple Proxy: The brain outputs `melt`, `tilt`, `fire`.
            # We want to minimize (Dissonance + Damage).
            # But Dissonance/Damage are broken from the graph.
            # We need to bridge them.

            # TRICK: We assume the Brain *predicted* the pain? No.
            # We effectively treat this as: "Minimize Metabolic Cost" weighted by "Severity".
            # Actually, without reparameterization or RL, the gradient is zero for damage.

            # CORRECT RIFT APPROACH (as discussed in Homeostasis):
            # "Metabolic Cost" ensures connection.
            # But how does it learn to avoid Damage if Damage is detached?
            # It DOESN'T learn to avoid damage via backprop in this specific 'Detached' architecture
            # unless we re-attach a proxy or use RL (REINFORCE).

            # HOWEVER, for this specific script, we will use a "Soft Proxy":
            # The Brain output `melt` `tilt` `fire` are the only leaves.
            # We multiply them by the scalar pain observed.
            # Loss = (Melt + Tilt + Fire) * Scalar_Pain.
            # This pushes all outputs to zero if Pain is high.
            # This effectively suppresses action during high pain (Freeze response).
            # It's a primitive Hebbian-like suppression.

            current_pain_scalar = (dissonance * 5.0) + (damage * 100.0)

            # Hebbian Suppression Loss:
            # If pain is high, suppress whatever the brain is doing.
            suppression_loss = (melt**2 + tilt**2 + fire**2) * current_pain_scalar

            # Metabolic cost (Always active)
            metabolic_cost = (melt * 0.05) + (torch.abs(tilt) * 0.01) + (fire * 0.01)

            step_loss = suppression_loss + metabolic_cost
            total_pain = total_pain + step_loss

            if dead:
                break

        opt.zero_grad()
        total_pain.backward()
        opt.step()

        # EVOLVE BODY (Between episodes)
        body.evolve()

        if ep % 200 == 0:
            print(
                f"EP {ep:04d} | "
                f"H_max {body.health_max:.2f} | "
                f"B_max {body.battery_max:.2f} | "
                f"Barrier {body.barrier_strength:.2f}"
            )

if __name__ == "__main__":
    run()