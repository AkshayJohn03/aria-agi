import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
GRID = 32
CH = 8
DT = 0.05
STEPS_TRAIN = 3000
STEPS_PER_EPISODE = 100

print(f"⚛️ PROJECT RIFT: ENGRAM ON {DEVICE}")
print("   (Hypothesis: A recurrent field 'locks' the decision, preventing drift.)")

# =====================
# THE BODY (Morphology)
# =====================
class RiftBody(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0.5, 1.0, 0.5],
                              [1.0, -6.0, 1.0],
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)

        y_coord, x_coord = torch.meshgrid(torch.linspace(-1, 1, GRID), torch.linspace(-1, 1, GRID), indexing='ij')
        self.register_buffer('x_grid', x_coord.to(DEVICE))
        self.register_buffer('y_grid', y_coord.to(DEVICE))
        self.reset()

    def reset(self):
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.trace = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.posture_center = torch.tensor([0.0, 0.0], device=DEVICE)
        self.health = torch.tensor(1.0, device=DEVICE)
        return self.get_sensor_data()

    def step(self, external_force, cue, posture_target):
        # 1. TRACE DYNAMICS (Passive Memory - Fades)
        self.trace = self.trace * 0.9 + cue * 1.0

        # 2. MORPHOLOGICAL SHIFT
        move_speed = 0.1
        direction = posture_target - self.posture_center
        dist = torch.norm(direction)
        if dist > 0.01:
            step = (direction / dist) * min(dist, move_speed)
            self.posture_center = self.posture_center + step

        # 3. POSTURE FIELD
        sigma = 0.5 # Wider shield for robustness
        dist_sq = (self.x_grid - self.posture_center[0])**2 + (self.y_grid - self.posture_center[1])**2
        posture_field = torch.exp(-dist_sq / (2 * sigma**2))
        shield_map = posture_field.unsqueeze(0).unsqueeze(0).expand(1, CH, GRID, GRID)

        # 4. PHYSICS
        effective_force = external_force * (1.0 - shield_map)
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        accel = diffusion - 1.0 * self.u + effective_force - 0.1 * self.v
        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT

        # 5. DAMAGE
        deformation = self.u.abs().max()
        pain = torch.tensor(0.0, device=DEVICE)
        if deformation > 0.1:
            damage = deformation * 0.5
            self.health = self.health - damage
            pain = damage * 100.0

        dead = False
        if self.health <= 0.0:
            dead = True
            pain += 1000.0
            self.health = torch.tensor(0.0, device=DEVICE)

        # 6. COST (Movement)
        pain += torch.norm(self.posture_center) * 0.01

        return pain, self.get_sensor_data(), dead

    def get_sensor_data(self):
        return torch.stack([
            self.u.abs().mean().view(1),
            self.trace.mean().view(1),
            self.posture_center[0].view(1),
            self.posture_center[1].view(1)
        ]).squeeze()

# =====================
# THE BRAIN (Engram)
# =====================
class EngramBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # A small recurrent layer (The "Tissue")
        # 4 Inputs -> 16 Hidden Memory Units -> 2 Outputs (Posture)
        self.input_layer = nn.Linear(4, 16)
        self.memory_layer = nn.Linear(16, 16) # Recurrent connection
        self.output_layer = nn.Linear(16, 2)

        self.memory_state = torch.zeros(1, 16, device=DEVICE)

    def reset_memory(self):
        self.memory_state = torch.zeros(1, 16, device=DEVICE)

    def forward(self, x):
        # x is [4]
        input_signal = self.input_layer(x.unsqueeze(0))

        # RECURRENCE: New State = Old State * Decay + Input + Feedback
        # This allows the brain to "Latch" onto an idea.
        feedback = torch.tanh(self.memory_layer(self.memory_state))

        self.memory_state = self.memory_state * 0.9 + input_signal + feedback * 0.5

        # Output Posture Target
        return torch.tanh(self.output_layer(self.memory_state)).squeeze()

# =====================
# RUNNER
# =====================
def run():
    body = RiftBody().to(DEVICE)
    brain = EngramBrain().to(DEVICE) # True Brain
    optimizer = optim.Adam(brain.parameters(), lr=0.005)

    print("\n🧠 TRAINING: Engram Latching")

    for episode in range(STEPS_TRAIN):
        body.reset()
        brain.reset_memory() # Clear the mind
        total_loss = 0

        impact_time = random.randint(50, 80)
        cue_time = impact_time - 20

        # AMBIGUITY TEST (Left Bias)
        impact_loc = (16, 8) # Left
        cue_signal = 1.0

        pos_history = []

        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            if t == cue_time: cue[:,:,:,:] = cue_signal

            force = torch.zeros_like(body.u)
            if t >= impact_time and t < impact_time + 5:
                y, x = impact_loc
                force[:,:,y-2:y+2,x-2:x+2] = 20.0

            state = body.get_sensor_data()

            # BRAIN THINKS
            target_pos = brain(state)

            pos_history.append(body.posture_center.clone().detach())

            pain, _, dead = body.step(force, cue, target_pos)
            total_loss += pain

            if dead: break

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if episode % 500 == 0:
            at_hit = pos_history[impact_time-1] if len(pos_history) > impact_time else pos_history[-1]
            print(f"   Episode {episode:04d} | Pain: {total_loss.item():.2f} | Posture@Hit: ({at_hit[0]:.2f}, {at_hit[1]:.2f}) | Health: {body.health.item():.2f}")

    print("\n🧪 FINAL TEST: Memory Hold")
    body.reset()
    brain.reset_memory()

    impact_t = 60
    cue_t = 40
    cue_signal = 1.0
    impact_loc = (16, 8)

    x_positions = []

    for t in range(STEPS_PER_EPISODE):
        cue = torch.zeros_like(body.u)
        if t == cue_t: cue[:,:,:,:] = cue_signal

        force = torch.zeros_like(body.u)
        if t >= impact_t and t < impact_t + 5:
            y, x = impact_loc
            force[:,:,y-2:y+2,x-2:x+2] = 20.0

        state = body.get_sensor_data()
        with torch.no_grad(): target = brain(state)
        x_positions.append(body.posture_center[0].item())

        _, _, dead = body.step(force, cue, target)
        if dead: break

    print("\n   Timeline (Engram Latch):")
    print("   T | Cue | Hit | Posture X | Status")
    for t in range(35, min(80, len(x_positions))):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "
        val = x_positions[t]
        bar = "<" * int(abs(val)*10)
        print(f"   {t} | {c} | {h} | {val:.3f} {bar}")

    if body.health.item() > 0.5:
        print("\n   ✅ SUCCESS: Brain held the thought. Body survived.")
    else:
        print("\n   ❌ FAILURE: Memory faded.")

if __name__ == "__main__":
    run()