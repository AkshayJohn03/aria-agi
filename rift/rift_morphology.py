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

print(f"🗿 PROJECT RIFT: MORPHOLOGY ON {DEVICE}")
print("   (Hypothesis: Survival requires pre-contact geometric commitment.)")

# =====================
# THE BODY (Morphological)
# =====================
class RiftMorphBody(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[[[0.5, 1.0, 0.5], 
                              [1.0, -6.0, 1.0], 
                              [0.5, 1.0, 0.5]]]], device=DEVICE)
        self.laplacian = lap.repeat(CH, 1, 1, 1)
        
        # Spatial coordinate grids for posture calculation
        y_coord, x_coord = torch.meshgrid(torch.linspace(-1, 1, GRID), torch.linspace(-1, 1, GRID), indexing='ij')
        self.register_buffer('x_grid', x_coord.to(DEVICE))
        self.register_buffer('y_grid', y_coord.to(DEVICE))
        
        self.reset()

    def reset(self):
        self.u = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.v = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        self.trace = torch.zeros(1, CH, GRID, GRID, device=DEVICE)
        
        # POSTURE STATE
        # Current center of mass (x, y) ranges from -1 to 1
        self.posture_center = torch.tensor([0.0, 0.0], device=DEVICE) 
        self.health = torch.tensor(1.0, device=DEVICE)
        
        return self.get_sensor_data()

    def step(self, external_force, cue, posture_target):
        # posture_target: Tensor [2] (Target X, Target Y)
        
        # 1. TRACE DYNAMICS (Memory)
        self.trace = self.trace * 0.9 + cue * 1.0
        
        # 2. MORPHOLOGICAL SHIFT (The "Heavy" Movement)
        # You cannot teleport center of mass. You must drag it.
        # This inertia is what forces anticipation.
        move_speed = 0.1 # Takes ~20 frames to cross the grid
        direction = posture_target - self.posture_center
        dist = torch.norm(direction)
        
        if dist > 0.01:
            step = (direction / dist) * min(dist, move_speed)
            self.posture_center = self.posture_center + step
            
        # 3. GENERATE POSTURE FIELD (The Shield)
        # Gaussian blob of defense centered at posture_center
        # Sigma defines how "focused" the stance is. Narrow = Strong but Risky.
        sigma = 0.4 
        dist_sq = (self.x_grid - self.posture_center[0])**2 + (self.y_grid - self.posture_center[1])**2
        posture_field = torch.exp(-dist_sq / (2 * sigma**2))
        
        # Normalize: Conservation of Mass. You can't be strong everywhere.
        # Peak strength depends on focus.
        posture_field = posture_field # Values 0.0 to 1.0 locally
        
        # 4. PHYSICS & GEOMETRIC IMPACT
        # The Shield is spatial. Force is multiplied by (1 - posture).
        # If posture is 1.0 at impact site, Force -> 0.
        # If posture is 0.0 at impact site, Force -> Full.
        
        # Expand posture to channel dim
        shield_map = posture_field.unsqueeze(0).unsqueeze(0).expand(1, CH, GRID, GRID)
        
        # INVERT: 1.0 means protected (force blocked)
        # So we multiply force by (1.0 - shield)
        # BUT: We add a penalty for "Wrong Stance".
        # If you are hit where shield is < 0.2, you take EXTRA damage (Brittle).
        effective_force = external_force * (1.0 - shield_map)
        
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        accel = diffusion - 1.0 * self.u + effective_force - 0.1 * self.v
        
        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT
        
        # 5. DAMAGE LOGIC
        deformation = self.u.abs().max()
        pain = torch.tensor(0.0, device=DEVICE)
        
        if deformation > 0.1:
            # Damage calculation
            damage = deformation * 0.5
            self.health = self.health - damage
            pain = damage * 100.0
            
        dead = False
        if self.health <= 0.0:
            dead = True
            pain = pain + 1000.0
            self.health = torch.tensor(0.0, device=DEVICE)
            
        # 6. METABOLIC COST
        # Holding a non-neutral posture costs energy (Balance)
        dist_from_neutral = torch.norm(self.posture_center)
        balance_cost = dist_from_neutral * 0.05
        pain = pain + balance_cost

        return pain, self.get_sensor_data(), dead

    def get_sensor_data(self):
        # Brain sees: [Vibration, Trace, CurrentPostureX, CurrentPostureY]
        return torch.stack([
            self.u.abs().mean().view(1),
            self.trace.mean().view(1),
            self.posture_center[0].view(1),
            self.posture_center[1].view(1)
        ]).squeeze()

# =====================
# CONTROLLER (Spatial Reflex)
# =====================
class SpinalReflex(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 Inputs -> 2 Outputs (Target X, Target Y)
        # Maps "Trace intensity" to "Spatial Position"
        self.reflex_arc = nn.Linear(4, 2) 

    def forward(self, x):
        # Output is Tanh (-1 to 1) for spatial coordinates
        return torch.tanh(self.reflex_arc(x))

# =====================
# RUNNER
# =====================
def run():
    body = RiftMorphBody().to(DEVICE)
    spine = SpinalReflex().to(DEVICE)
    optimizer = optim.Adam(spine.parameters(), lr=0.01)
    
    print("\n🗿 TRAINING: Morphological Commitment")
    print("   (Hypothesis: You must shift your center of mass BEFORE impact.)")
    
    for episode in range(STEPS_TRAIN):
        body.reset()
        total_loss = 0
        
        impact_time = random.randint(50, 80)
        cue_time = impact_time - 20 # Give enough time to shift weight
        
        # Randomize Impact Location (Left or Right)
        # To force the brain to choose a side based on the Cue
        if random.random() > 0.5:
            impact_loc = (16, 8) # Left
            cue_signal = 1.0     # Positive signal = Left
        else:
            impact_loc = (16, 24) # Right
            cue_signal = -1.0     # Negative signal = Right
            
        pos_history = []
        
        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            # Spatial Cue: We inject the signal globally or locally
            if t == cue_time: 
                # We use the Trace channel to encode the "Side"
                # A positive trace means Left, Negative means Right (simplified for linear brain)
                cue[:,:,:,:] = cue_signal 
            
            force = torch.zeros_like(body.u)
            if t >= impact_time and t < impact_time + 5: 
                # Hammer hits specific spot
                y, x = impact_loc
                force[:,:,y-2:y+2,x-2:x+2] = 20.0 
            
            state = body.get_sensor_data()
            target_pos = spine(state).squeeze()
            
            pos_history.append(body.posture_center.clone().detach())
            
            pain, _, dead = body.step(force, cue, target_pos)
            total_loss += pain
            
            if dead: break
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if episode % 500 == 0:
            # Check where the body was at impact
            at_impact = pos_history[impact_time-1] if len(pos_history) > impact_time else pos_history[-1]
            print(f"   Episode {episode:04d} | Pain: {total_loss.item():.2f} | Posture@Hit: ({at_impact[0]:.2f}, {at_impact[1]:.2f}) | Health: {body.health.item():.2f}")

    print("\n🧪 FINAL TEST: The Dodge")
    # Test Left Impact
    body.reset()
    impact_t = 60
    cue_t = 40
    
    # SETUP LEFT IMPACT
    impact_loc = (16, 8) # Left (Y=16, X=8) -> X coord is approx -0.5
    cue_signal = 1.0
    
    x_positions = []
    
    for t in range(STEPS_PER_EPISODE):
        cue = torch.zeros_like(body.u)
        if t == cue_t: cue[:,:,:,:] = cue_signal
        
        force = torch.zeros_like(body.u)
        if t >= impact_t and t < impact_t + 5: 
            y, x = impact_loc
            force[:,:,y-2:y+2,x-2:x+2] = 20.0
        
        state = body.get_sensor_data()
        with torch.no_grad(): target = spine(state).squeeze()
        
        x_positions.append(body.posture_center[0].item())
        
        _, _, dead = body.step(force, cue, target)
        if dead: break

    print("\n   Timeline (Left Impact Test):")
    print("   T | Cue | Hit | Posture X (Should go Negative/Left) | Status")
    
    for t in range(35, min(80, len(x_positions))):
        c = "CUE" if t == cue_t else " . "
        h = "HIT" if t >= impact_t and t < impact_t+5 else " . "
        
        # Visualize Shift
        val = x_positions[t]
        bar_len = int(abs(val) * 10)
        bar = "<" * bar_len if val < 0 else ">" * bar_len
        
        print(f"   {t} | {c} | {h} | {val:.3f} {bar}")

    if body.health.item() > 0.5:
        print("\n   ✅ SUCCESS: Body shifted weight to blocked impact.")
    else:
        print("\n   ❌ FAILURE: Body did not move fast enough or correctly.")

if __name__ == "__main__":
    run()