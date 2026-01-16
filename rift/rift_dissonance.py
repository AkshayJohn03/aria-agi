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
STEPS_TRAIN = 5000 
STEPS_PER_EPISODE = 120

print(f"🔥 PROJECT RIFT: DISSONANCE (FIXED) ON {DEVICE}")
print("   (Hypothesis: Pain arises from the mismatch between Belief and Reality.)")

# =====================
# THE BODY (Thermodynamic + Capacitor + Dissonance)
# =====================
class RiftDissonanceBody(nn.Module):
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
        self.battery = torch.tensor(0.5, device=DEVICE) 
        
        return self.get_sensor_data()

    def step(self, external_force, cue, barrier_control, tilt_bias, fire_cmd):
        # 1. TRACE DYNAMICS
        self.trace = self.trace * 0.8 + cue * 1.0
        
        # 2. CALCULATE DISSONANCE (The New Force)
        belief = self.trace.mean()
        reality = self.posture_center[0]
        
        # Mismatch check
        alignment = torch.sign(belief) * torch.sign(reality)
        
        dissonance_pain = torch.tensor(0.0, device=DEVICE)
        
        if torch.abs(belief) > 0.1: 
            if alignment < 0: # Disagreement!
                # FIX: Ensure this remains a Tensor
                dissonance_pain = torch.tensor(0.5, device=DEVICE) 
                self.health = self.health - (dissonance_pain * 0.05) 
        
        # 3. CAPACITOR
        charge_rate = 0.02 
        self.battery = torch.clamp(self.battery + charge_rate, 0.0, 1.0)
        
        impulse_force = torch.tensor([0.0, 0.0], device=DEVICE)
        did_fire = 0.0
        
        if fire_cmd > 0.5 and self.battery > 0.5:
            did_fire = 1.0
            kick_strength = 6.0 
            direction = torch.sign(tilt_bias) 
            impulse_force[0] = direction * kick_strength * self.battery
            self.battery = torch.tensor(0.0, device=DEVICE)
            
        # 4. LANDSCAPE PHYSICS
        base_barrier = 2.0
        current_barrier = base_barrier * (1.0 - barrier_control) 
        
        px = self.posture_center[0]
        target_well = 0.6
        force_landscape = -4.0 * current_barrier * px * (px**2 - target_well**2)
        
        force_tilt = tilt_bias * 0.5 
        
        total_force_x = force_landscape + force_tilt + impulse_force[0]
        total_force_y = -2.0 * self.posture_center[1] 
        
        damping = 0.5 
        
        # Out-of-place updates
        new_vel_x = self.posture_vel[0] + (total_force_x - self.posture_vel[0] * damping) * DT
        new_vel_y = self.posture_vel[1] + (total_force_y - self.posture_vel[1] * damping) * DT
        self.posture_vel = torch.stack([new_vel_x, new_vel_y])
        
        self.posture_center = self.posture_center + self.posture_vel * DT
        self.posture_center = torch.clamp(self.posture_center, -1.0, 1.0)
        
        # 5. POSTURE FIELD
        sigma = 0.4
        dist_sq = (self.x_grid - self.posture_center[0])**2 + (self.y_grid - self.posture_center[1])**2
        posture_field = torch.exp(-dist_sq / (2 * sigma**2))
        shield_map = posture_field.unsqueeze(0).unsqueeze(0).expand(1, CH, GRID, GRID)
        
        # 6. PHYSICS
        effective_force = external_force * (1.0 - shield_map)
        diffusion = F.conv2d(self.u, self.laplacian, padding=1, groups=CH)
        
        accel = diffusion - 1.0 * self.u + effective_force - 0.1 * self.v
        self.v = self.v + accel * DT
        self.u = self.u + self.v * DT
        
        # 7. DAMAGE
        deformation = self.u.abs().max()
        pain = torch.tensor(0.0, device=DEVICE)
        
        if deformation > 0.1:
            damage = deformation * 0.5
            self.health = self.health - damage
            pain = damage * 100.0
            
        # Dissonance Pain (Feedback to brain)
        pain = pain + (dissonance_pain * 10.0)
            
        dead = False
        if self.health <= 0.0:
            dead = True
            pain = pain + 1000.0
            self.health = torch.tensor(0.0, device=DEVICE)

        return pain, self.get_sensor_data(), dead, did_fire, dissonance_pain

    def get_sensor_data(self):
        return torch.stack([
            self.u.abs().mean().view(1),
            self.trace.mean().view(1),
            self.posture_center[0].view(1),
            self.posture_center[1].view(1),
            self.battery.view(1),
            self.health.view(1)
        ]).squeeze()

# =====================
# THE CONTROLLER
# =====================
class SpinalReflex(nn.Module):
    def __init__(self):
        super().__init__()
        self.reflex_arc = nn.Linear(6, 3) 

    def forward(self, x):
        out = self.reflex_arc(x)
        melt = torch.sigmoid(out[0]) 
        tilt = torch.tanh(out[1])
        fire = torch.sigmoid(out[2]) 
        return melt, tilt, fire

# =====================
# RUNNER
# =====================
def run():
    body = RiftDissonanceBody().to(DEVICE)
    spine = SpinalReflex().to(DEVICE)
    optimizer = optim.Adam(spine.parameters(), lr=0.01)
    
    print("\n🔥 TRAINING: Cognitive Dissonance")
    
    for episode in range(STEPS_TRAIN):
        body.reset()
        total_loss = 0
        
        impact_time = random.randint(80, 110)
        cue_time_1 = impact_time - 60 
        cue_time_2 = impact_time - 20 
        
        if random.random() > 0.5:
            c1, c2 = 1.0, -1.0
            impact_loc = (16, 24) # Right
        else:
            c1, c2 = -1.0, 1.0
            impact_loc = (16, 8) # Left
            
        pos_history = []
        fire_history = []
        dissonance_history = []
        
        for t in range(STEPS_PER_EPISODE):
            cue = torch.zeros_like(body.u)
            if t == cue_time_1: cue[:,:,:,:] = c1
            if t == cue_time_2: cue[:,:,:,:] = c2 * 2.0 
            
            force = torch.zeros_like(body.u)
            if t >= impact_time and t < impact_time + 5: 
                y, x = impact_loc
                force[:,:,y-2:y+2,x-2:x+2] = 20.0 
            
            state = body.get_sensor_data()
            melt, tilt, fire = spine(state)
            
            pos_history.append(body.posture_center[0].item())
            
            pain, _, dead, did_fire, diss = body.step(force, cue, melt, tilt, fire)
            total_loss += pain
            
            if did_fire > 0.5:
                fire_history.append(t)
            dissonance_history.append(diss.item())
            
            if dead: break
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if episode % 500 == 0:
            at_hit = pos_history[impact_time-1] if len(pos_history) > impact_time else pos_history[-1]
            fired = "YES" if len(fire_history) > 0 else "NO"
            avg_diss = sum(dissonance_history)/len(dissonance_history) if len(dissonance_history)>0 else 0
            print(f"   Episode {episode:04d} | HitPos: {at_hit:.2f} | Fired: {fired} | Dissonance: {avg_diss:.2f} | Health: {body.health.item():.2f}")

    print("\n🧪 FINAL TEST: The Burning Need to Switch")
    body.reset()
    
    cue_t1 = 20
    cue_t2 = 60
    impact_t = 100
    
    c1, c2 = 1.0, -1.0 
    impact_loc = (16, 24) # Right
    
    x_positions = []
    fires = []
    pain_levels = []
    
    for t in range(STEPS_PER_EPISODE):
        cue = torch.zeros_like(body.u)
        if t == cue_t1: cue[:,:,:,:] = c1
        if t == cue_t2: cue[:,:,:,:] = c2 * 2.0
        
        force = torch.zeros_like(body.u)
        if t >= impact_t and t < impact_t + 5: 
            y, x = impact_loc
            force[:,:,y-2:y+2,x-2:x+2] = 20.0
        
        state = body.get_sensor_data()
        with torch.no_grad(): melt, tilt, fire = spine(state)
        
        x_positions.append(body.posture_center[0].item())
        
        _, _, dead, did_fire, diss = body.step(force, cue, melt, tilt, fire)
        fires.append(did_fire)
        pain_levels.append(diss.item())
        
        if dead: break

    print("\n   Timeline (Dissonance):")
    print("   T | Event | Posture X | Dissonance | Action")
    
    for t in range(15, min(110, len(x_positions))):
        evt = " . "
        if t == cue_t1: evt = "C1 "
        if t == cue_t2: evt = "C2 "
        if t >= impact_t and t < impact_t+5: evt = "HIT"
        
        pos = x_positions[t]
        diss = pain_levels[t]
        fired = "BOOM" if fires[t] > 0 else "    "
        
        bar_pos = "<" * int(abs(pos)*10) if pos < 0 else ">" * int(abs(pos)*10)
        bar_dis = "!" * int(diss*10)
        
        print(f"   {t} | {evt} | {pos:.3f} {bar_pos:10} | {diss:.1f} {bar_dis} | {fired}")

    if x_positions[impact_t-1] > 0.5 and body.health.item() > 0.2:
        print("\n   ✅ SUCCESS: Dissonance forced the Impulse. System Switched.")
    else:
        print("\n   ❌ FAILURE: Did not switch or died from burnout.")

if __name__ == "__main__":
    run()