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

print(f"🔒 PROJECT RIFT: HYSTERESIS ON {DEVICE}")
print("   (Hypothesis: Friction creates memory. The body holds its shape until forced otherwise.)")

# =====================
# THE BODY (Hysteresis)
# =====================
class RiftHysteresisBody(nn.Module):
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
        
        # POSTURE STATE
        self.posture_center = torch.tensor([0.0, 0.0], device=DEVICE) 
        self.health = torch.tensor(1.0, device=DEVICE)
        
        return self.get_sensor_data()

    def step(self, external_force, cue, posture_push):
        # 1. TRACE DYNAMICS (Decays normally)
        self.trace = self.trace * 0.9 + cue * 1.0
        
        # 2. HYSTERESIS MOVEMENT (The Fix)
        # We model the movement as pushing a heavy block with friction.
        
        # posture_push is the force from the reflex (-1 to 1)
        # FRICTION THRESHOLD: You must push hard to move.
        static_friction = 0.1
        
        push_magnitude = torch.norm(posture_push)
        move_vector = torch.zeros_like(self.posture_center)
        
        if push_magnitude > static_friction:
            # You overcame friction!
            # Movement is proportional to (Push - Friction)
            # We REMOVE the "Elasticity" term that pulled it back to 0,0.
            # The body has NO desire to center itself.
            move_speed = 0.05 
            direction = posture_push / push_magnitude
            move_vector = direction * move_speed
            
        self.posture_center = self.posture_center + move_vector
        
        # Clamp to grid limits
        self.posture_center = torch.clamp(self.posture_center, -1.0, 1.0)
            
        # 3. POSTURE FIELD
        sigma = 0.4 
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

        # 6. COST
        # Cost of PUSHING, not cost of HOLDING.
        # This rewards "Move once, then stop."
        metabolic_cost = torch.norm(move_vector) * 1.0
        pain += metabolic_cost

        return pain, self.get_sensor_data(), dead

    def get_sensor_data(self):
        return torch.stack([
            self.u.abs().mean().view(1),
            self.trace.mean().view(1),
            self.posture_center[0].view(1),
            self.posture_center[1].view(1)
        ]).squeeze()

# =====================
# THE CONTROLLER (Back to Reflex)
# =====================
class SpinalReflex(nn.Module):
    def __init__(self):
        super().__init__()
        # We are back to a simple, stateless reflex.
        # The "Memory" is now in the Body's friction.
        self.reflex_arc = nn.Linear(4, 2) 

    def forward(self, x):
        return torch.tanh(self.reflex_arc(x))

# =====================
# RUNNER
# =====================
def run():
    body = RiftHysteresisBody().to(DEVICE)
    spine = SpinalReflex().to(DEVICE)
    optimizer = optim.Adam(spine.parameters(), lr=0.01)
    
    print("\n🔒 TRAINING: Physical Hysteresis")
    
    for episode in range(STEPS_TRAIN):
        body.reset()
        total_loss = 0
        
        impact_time = random.randint(50, 80)
        cue_time = impact_time - 25 # Long gap to test memory
        
        # Ambiguity: 1.0 Cue -> Left Hit
        cue_signal = 1.0
        impact_loc = (16, 8)
        
        pos_history = []
        
        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            if t == cue_time: cue[:,:,:,:] = cue_signal 
            
            force = torch.zeros_like(body.u)
            if t >= impact_time and t < impact_time + 5: 
                y, x = impact_loc
                force[:,:,y-2:y+2,x-2:x+2] = 20.0 
            
            state = body.get_sensor_data()
            target_push = spine(state).squeeze() # Output is Force, not Position
            
            pos_history.append(body.posture_center.clone().detach())
            
            pain, _, dead = body.step(force, cue, target_push)
            total_loss += pain
            
            if dead: break
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if episode % 500 == 0:
            at_hit = pos_history[impact_time-1] if len(pos_history) > impact_time else pos_history[-1]
            print(f"   Episode {episode:04d} | Pain: {total_loss.item():.2f} | Posture@Hit: ({at_hit[0]:.2f}, {at_hit[1]:.2f}) | Health: {body.health.item():.2f}")

    print("\n🧪 FINAL TEST: Friction Hold")
    body.reset()
    
    impact_t = 70 # Very late hit
    cue_t = 30 # Early cue
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
        with torch.no_grad(): push = spine(state).squeeze()
        x_positions.append(body.posture_center[0].item())
        
        _, _, dead = body.step(force, cue, push)
        if dead: break

    print("\n   Timeline (Hysteresis):")
    print("   T | Cue | Hit | Posture X (Should stay negative) | Status")
    for t in range(25, min(80, len(x_positions))):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "
        val = x_positions[t]
        bar = "<" * int(abs(val)*10)
        print(f"   {t} | {c} | {h} | {val:.3f} {bar}")

    if body.health.item() > 0.5 and x_positions[-1] < -0.4:
        print("\n   ✅ SUCCESS: Body remembered via Friction.")
    else:
        print("\n   ❌ FAILURE: Body drifted.")

if __name__ == "__main__":
    run()