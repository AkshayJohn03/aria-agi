import torch
import torch.nn as nn
import torch.nn.functional as F
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32
CH = 8
DT = 0.05
STEPS_PER_EPISODE = 100

class RiftWorldCommitment(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        # Laplacian kernel for diffusion
        lap = torch.tensor([[[[0.5, 1.0, 0.5],
                              [1.0, -6.0, 1.0],
                              [0.5, 1.0, 0.5]]]], device=self.device)
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        # State
        self.u = None
        self.v = None
        self.trace_fast = None
        self.trace_slow = None
        self.brace_charge = None
        self.health = None
        self.is_committed = False

        # Episode Config
        self.hazard_time = -1
        self.hazard_active = False

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.u = torch.zeros(1, CH, GRID, GRID, device=self.device)
        self.v = torch.zeros(1, CH, GRID, GRID, device=self.device)
        self.trace_fast = torch.zeros(1, CH, GRID, GRID, device=self.device)
        self.trace_slow = torch.zeros(1, CH, GRID, GRID, device=self.device)

        self.brace_charge = torch.tensor(0.0, device=self.device)
        self.health = torch.tensor(1.0, device=self.device)
        self.is_committed = False

        # Determine Hazard Timing (Deterministic based on seed if provided)
        # Hazard happens between step 40 and 80
        self.hazard_time = random.randint(40, 80)
        self.cue_time = self.hazard_time - 15

        return self.get_observation_4d()

    def step(self, action_commit: bool, step_idx: int):
        """
        action_commit: Boolean. If True, attempts to commit.
        """
        # --- LATCHING LOGIC (Irreversibility) ---
        if action_commit and not self.is_committed:
            self.is_committed = True

        # If committed, brace is LOCKED at 1.0
        # If not, brace is 0.0 (Relaxed)
        # (The original allowed gradual charge, here we force binary commitment)
        target_charge = 1.0 if self.is_committed else 0.0

        # Physical Charge Dynamics (still has inertia)
        # If we commit, it ramps up fast.
        if self.is_committed:
            self.brace_charge = self.brace_charge * 0.8 + target_charge * 0.2
        else:
            self.brace_charge = self.brace_charge * 0.8 # Decay if not committed

        self.brace_charge = torch.clamp(self.brace_charge, 0.0, 1.0)

        # --- WORLD EVENTS ---
        # Cue (Trace signal)
        # Increase cue intensity significantly (1.0 -> 5.0) to make it visible through noise
        cue = torch.zeros_like(self.u)
        if step_idx == self.cue_time:
            cue[:,:,16-4:16+4,16-4:16+4] = 5.0

        # Hazard (Force)
        force = torch.zeros_like(self.u)
        hazard_active = (step_idx >= self.hazard_time and step_idx < self.hazard_time + 5)
        if hazard_active:
            force[:,:,16-2:16+2,16-2:16+2] = 20.0 # Lethal impact

        # --- PHYSICS UPDATE ---
        # 1. Memory Traces
        # CALIBRATION: Decay 0.8 -> 0.9 to extend causal window
        self.trace_fast = self.trace_fast * 0.9 + cue * 1.0
        self.trace_slow = self.trace_slow * 0.98 + cue * 0.5

        # 2. Ischemia (Cost of Commitment)
        # Holding tension > 0.3 rots the body
        ischemic_damage = torch.tensor(0.0, device=self.device)
        if self.brace_charge > 0.3:
            # CALIBRATION: Cost 0.05 -> 0.03 to allow 15-step hold
            ischemic_damage = (self.brace_charge - 0.3) * 0.03
            self.health = self.health - ischemic_damage

        # 3. Shielding & Diffusion
        shield = (1.0 - self.brace_charge) # 1.0 = Exposed, 0.0 = Protected

        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        restoring = -1.0 * self.u
        accel = diffusion + restoring + (force * shield) - 0.1 * self.v

        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 4. Trauma (Damage from deformation)
        deformation = self.u.abs().max()
        if deformation > 0.1:
            trauma = deformation * 0.2
            self.health = self.health - trauma

        # 5. Mortality
        done = False
        reward = 0.0

        if self.health <= 0.0:
            self.health = torch.tensor(0.0, device=self.device)
            done = True
            reward = -1.0 # Death Penalty

        # End of Episode Check
        if step_idx >= STEPS_PER_EPISODE - 1 and not done:
            done = True
            reward = 1.0 + (self.health.item() * 0.5) # Survival Bonus + Efficiency

        return self.get_observation_4d(), reward, done, {
            "health": self.health.item(),
            "brace": self.brace_charge.item(),
            "committed": self.is_committed,
            "hazard_active": hazard_active,
            "deformation": deformation.item()
        }

    def get_observation_4d(self):
        """
        Returns [u_mean, trace_fast_mean, brace, health]
        """
        # Noise Logic (Blindness): High Tension = High Noise
        noise_level = self.brace_charge * 1.0

        raw_u = self.u.abs().mean().view(1)
        raw_trace = self.trace_fast.mean().view(1)

        # Apply noise
        # Reduced noise level on trace to ensure visibility
        noisy_u = raw_u + torch.randn_like(raw_u) * noise_level * 0.1
        noisy_trace = raw_trace + torch.randn_like(raw_trace) * noise_level * 0.2 # Was 0.5

        return torch.stack([
            noisy_u,
            noisy_trace,
            self.brace_charge.view(1),
            self.health.view(1)
        ]).squeeze()

    def get_observation_grid(self):
        """
        Returns the raw 32x32 grid (Phase A2).
        Includes U and Trace channels.
        """
        # Concatenate U and Trace
        return torch.cat([self.u, self.trace_fast], dim=1)
