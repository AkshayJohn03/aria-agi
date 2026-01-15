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

print(f"🌋 PROJECT RIFT: PHASE TRANSITION ON {DEVICE}")
print("   (Hypothesis: Memory is trapped by raising/lowering energy barriers.)")

# =====================
# THE BODY (Thermodynamic)
# =====================
class RiftPhaseBody(nn.Module):
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
        self.posture_vel = torch.tensor([0.0, 0.0], device=DEVICE)
        self.health = torch.tensor(1.0, device=DEVICE)
        return self.get_sensor_data()

    def step(self, external_force, cue, barrier_control, tilt_bias):
        # barrier_control: 0.0 (Wall Up) to 1.0 (Wall Melted)
        # tilt_bias: -1.0 to 1.0 (Slight bias to guide the fall)
        
        # 1. TRACE
        self.trace = self.trace * 0.9 + cue * 1.0
        
        # 2. PHASE TRANSITION PHYSICS
        # Potential V(x) = Barrier * (1 - x^2)^2 - Tilt * x
        # Force = -dV/dx
        
        # BARRIER HEIGHT:
        # Default = 2.0 (High Wall). Brain can lower it to 0.0.
        base_barrier = 2.0
        current_barrier = base_barrier * (1.0 - barrier_control) 
        
        px = self.posture_center[0]
        
        # Force from the Landscape
        # If Barrier is High: Strong restoring force to 0 or +/- 1
        # If Barrier is Low: It becomes a flat floor or single well
        force_landscape = -4.0 * current_barrier * px * (px**2 - 1.0)
        
        # Force from Tilt (The Nudge)
        # We only need a tiny nudge if the wall is melted
        force_tilt = tilt_bias * 0.5 
        
        total_force_x = force_landscape + force_tilt
        total_force_y = -2.0 * self.posture_center[1] # Keep Y centered
        
        # 3. DAMPING (Energy Dissipation)
        # Critical: Wells must capture energy. Damping is high inside wells.
        damping = 0.5 
        
        # Out-of-place updates
        new_vel_x = self.posture_vel[0] + (total_force_x - self.posture_vel[0] * damping) * DT
        new_vel_y = self.posture_vel[1] + (total_force_y - self.posture_vel[1] * damping) * DT
        self.posture_vel = torch.stack([new_vel_x, new_vel_y])
        
        self.posture_center = self.posture_center + self.posture_vel * DT
        self.posture_center = torch.clamp(self.posture_center, -0.9, 0.9)
        
        # 4. POSTURE FIELD
        sigma = 0.4
        dist_sq = (self.x_grid - self.posture_center[0])**2 + (self.y_grid - self.posture_center[1])**2
        posture_field = torch.exp(-dist_sq / (2 * sigma**2))
        shield_map = posture_field.unsqueeze(0).unsqueeze(0).expand(1, CH, GRID, GRID)
        
        # 5. PHYSICS
        effective_force = external_force * (1.0 - shield_map)
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        accel = diffusion - 1.0 * self.u + effective_force - 0.1 * self.v
        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT
        
        # 6. DAMAGE
        deformation = self.u.abs().max()
        pain = torch.tensor(0.0, device=DEVICE)
        if deformation > 0.1:
            damage = deformation * 0.5
            self.health = self.health - damage
            pain = damage * 100.0
            
        dead = False
        if self.health <= 0.0:
            dead = True
            pain = pain + 1000.0
            self.health = torch.tensor(0.0, device=DEVICE)

        # 7. METABOLIC COST
        # CRITICAL: You pay for MELTING THE WALL (barrier_control).
        # You do NOT pay for existing (posture state).
        thermo_cost = barrier_control * 0.5
        pain = pain + thermo_cost

        return pain, self.get_sensor_data(), dead

    def get_sensor_data(self):
        return torch.stack([
            self.u.abs().mean().view(1),
            self.trace.mean().view(1),
            self.posture_center[0].view(1),
            self.posture_center[1].view(1)
        ]).squeeze()

# =====================
# THE CONTROLLER
# =====================
class SpinalReflex(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 Inputs -> 2 Outputs (Melt Wall, Tilt Bias)
        self.reflex_arc = nn.Linear(4, 2) 

    def forward(self, x):
        out = self.reflex_arc(x)
        # Melt: Sigmoid (0 to 1) - Can't be negative
        melt = torch.sigmoid(out[0]) 
        # Tilt: Tanh (-1 to 1) - Can push Left or Right
        tilt = torch.tanh(out[1])
        return melt, tilt

# =====================
# RUNNER
# =====================
def run():
    body = RiftPhaseBody().to(DEVICE)
    spine = SpinalReflex().to(DEVICE)
    optimizer = optim.Adam(spine.parameters(), lr=0.01)
    
    print("\n🌋 TRAINING: Phase Transition Memory")
    
    for episode in range(STEPS_TRAIN):
        body.reset()
        total_loss = 0
        
        impact_time = random.randint(50, 80)
        cue_time = impact_time - 30 # Huge gap
        
        cue_signal = 1.0 # Left Bias
        impact_loc = (16, 8) 
        
        pos_history = []
        effort_history = []
        
        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            if t == cue_time: cue[:,:,:,:] = cue_signal 
            
            force = torch.zeros_like(body.u)
            if t >= impact_time and t < impact_time + 5: 
                y, x = impact_loc
                force[:,:,y-2:y+2,x-2:x+2] = 20.0 
            
            state = body.get_sensor_data()
            melt, tilt = spine(state)
            
            pos_history.append(body.posture_center[0].item())
            effort_history.append(melt.item())
            
            pain, _, dead = body.step(force, cue, melt, tilt)
            total_loss += pain
            
            if dead: break
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if episode % 500 == 0:
            at_hit = pos_history[impact_time-1] if len(pos_history) > impact_time else pos_history[-1]
            print(f"   Episode {episode:04d} | Pain: {total_loss.item():.2f} | Posture@Hit: {at_hit:.2f} | Health: {body.health.item():.2f}")

    print("\n🧪 FINAL TEST: Thermodynamic Lock")
    body.reset()
    
    impact_t = 80 
    cue_t = 20 
    cue_signal = 1.0
    impact_loc = (16, 8)
    
    x_positions = []
    energy_burn = [] # Tracks "Melt" effort
    
    for t in range(STEPS_PER_EPISODE):
        cue = torch.zeros_like(body.u)
        if t == cue_t: cue[:,:,:,:] = cue_signal
        
        force = torch.zeros_like(body.u)
        if t >= impact_t and t < impact_t + 5: 
            y, x = impact_loc
            force[:,:,y-2:y+2,x-2:x+2] = 20.0
        
        state = body.get_sensor_data()
        with torch.no_grad(): melt, tilt = spine(state)
        
        x_positions.append(body.posture_center[0].item())
        energy_burn.append(melt.item())
        
        _, _, dead = body.step(force, cue, melt, tilt)
        if dead: break

    print("\n   Timeline (Phase Transition):")
    print("   T | Cue | Hit | Posture (Target -0.9) | Barrier Melt (Cost) | Status")
    
    for t in range(15, min(90, len(x_positions))):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "
        
        pos = x_positions[t]
        effort = energy_burn[t]
        
        bar_pos = "<" * int(abs(pos)*10)
        bar_eff = "!" * int(effort*10)
        
        print(f"   {t} | {c} | {h} | {pos:.3f} {bar_pos:10} | {effort:.3f} {bar_eff}")

    # SUCCESS CRITERIA:
    # 1. Posture Locked (< -0.8)
    # 2. Effort Zero after Cue (t=40+)
    
    hold_effort = sum(energy_burn[40:70]) / 30.0
    final_pos = x_positions[impact_t-1]
    
    if final_pos < -0.8 and hold_effort < 0.1:
        print("\n   🏆 GLORIOUS SUCCESS: Posture Trapped. Zero Cost. True Memory.")
    elif final_pos < -0.8:
        print("\n   ⚠️ PARTIAL: Locked, but Brain kept melting the wall.")
    else:
        print("\n   ❌ FAILURE: Did not commit.")

if __name__ == "__main__":
    run()