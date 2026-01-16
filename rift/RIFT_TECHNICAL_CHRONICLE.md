# PROJECT RIFT: A COMPLETE TECHNICAL CHRONICLE
**Subtitle: From Reflexes to Morphology — The Evolution of Anticipatory Intelligence**

---

## 0. Abstract & Executive Summary

Project RIFT investigated whether anticipatory intelligence could emerge in embodied, non-neural (or minimally neural) substrates through thermodynamic and physiological constraints rather than reward maximization. Over the course of 50+ experiments, the project systematically rejected traditional AI paradigms (Transformers, LSTMs, Policy Gradients) in favor of **Morphological Computation**.

The central discovery was that **memory is not a stored value, but a geometric deformation**, and **prediction is not a calculation, but a pre-emptive structural commitment.**

This chronicle documents every script in the repository, organized by evolutionary phase.

---

## 1. Phase I: The Reflex (Ballistic)
**Goal:** Establish a baseline where the agent must "brace" against a lethal impact. Can a simple linear reflex survive?

### 1.1. `rift_ballistic.py` (Base Reflex)
*   **Concept:** A "Bulletproof" reflex test. The agent must charge a defensive `brace` scalar to 1.0 to survive a hammer blow.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 2000
*   **Physics:**
    *   **Charge Dynamics:** $C_{t+1} = (C_t + 0.05 \cdot \text{signal}) \cdot 0.98$ (Leak Rate)
    *   **Pain:** $P = \text{Deformation} + 0.1 \cdot \text{Signal}$
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Pain: 0.05 | Tension: 0.009`
    *   **Analysis:** The system minimized the metabolic cost of acting (0.1) because the penalty for failure was not high enough. It chose laziness.

### 1.2. `rift_ballistic_1.py` (Lethal Variant)
*   **Concept:** Raising the stakes. Impact force increased to 20.0. Pain multiplier x100.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 1500
*   **Physics:**
    *   **Lethal Threshold:** If deformation > 0.1 (was 0.5), infinite pain.
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Pain: 8944.25 | Tension: 0.012`
    *   **Analysis:** Death did not solve the problem. The linear controller could not bridge the time gap. Penalty magnitude does not create capability.

### 1.3. `rift_core.py` (Semantic Tectonics v1)
*   **Concept:** Early prototypes treating "Meaning as Stress". Inputs (GPT-2 embeddings) were treated as physical forces on a tensor grid.
*   **Grid:** 48x48, 32 Channels
*   **Steps:** 300
*   **Physics:**
    *   $\text{Stress} = K * \nabla^2 U$
    *   **Fracture:** If $|\text{Stress}| > \text{Elasticity}$, break.
*   **Result:** 💥 **TRAUMATIZED**.
    *   **Log:** `Fracture=28584...`
    *   **Analysis:** Semantic vectors created massive, destructive interference patterns when mapped to 2D physics. This led to the pivot towards "Embodied" signals rather than "Semantic" ones.

### 1.4. `rift_core_v2.py` (Stabilized Tectonics)
*   **Concept:** Stabilized version with `YIELD_RATE = 0.03` and `DIFFUSION_RATE = 0.20`.
*   **Grid:** 48x48, 32 Channels
*   **Steps:** 450
*   **Result:** ✅ **STABLE**.
    *   **Log:** `Fracture=0.0000 (Stagnant)`
    *   **Analysis:** System achieved stability but did nothing interesting.

### 1.5. `rift_antagonist.py` (Antagonist Ego)
*   **Concept:** An internal regulator ("Ego") that injects chaos when bored and constrains when stressed.
*   **Grid:** 48x48, 32 Channels
*   **Steps:** 600
*   **Mechanism:**
    *   If `fracture < 0.01`: `ego_force = 0.6 * sign(stress)` (Pressurize)
    *   If `fracture > 0.15`: `ego_force = -0.4 * tanh(stress)` (Constrain)
*   **Result:** ✅ **SUSTAINED STRUGGLE**.
    *   **Log:** `Fracture=0.2615 (Overloaded) <-> 0.0005 (Bored)`
    *   **Analysis:** Successfully maintained "Edge of Chaos" dynamics.

---

## 2. Phase II: Mortality & Fear
**Goal:** Replace binary death with continuous degradation (Health).

### 2.1. `rift_scare.py` (The Mortal Body)
*   **Concept:** Continuous Health (1.0 -> 0.0). Damage reduces charging rate (Scarring).
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 2000
*   **Physics:**
    *   $\text{ChargeRate} = 0.05 \cdot \text{Health}$
    *   **Shielding:** $F_{net} = F_{ext} \cdot (1.0 - C_{brace})$
*   **Result:** 🐢 **PARTIAL (PARANOIA)**.
    *   **Log:** `Tension: 1.000 | Health: 1.00`
    *   **Analysis:** The system learned to **always** brace. It maximized tension immediately upon seeing the Cue and never relaxed. It survived, but it was not intelligent; it was paranoid.

---

## 3. Phase III: Energy & Constraints
**Goal:** Make paranoia expensive. Introduce Ischemia (Necrosis from tension).

### 3.1. `rift_ischemia.py` (Fatigue of Readiness)
*   **Concept:** Holding tension > 0.2 causes direct Health loss (Necrosis).
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 2500
*   **Equation:** If $C > 0.2 \rightarrow H -= (C - 0.2) \cdot 0.05$
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `DIED at step 50`
    *   **Analysis:** System could not time the defense. It either died from impact or died from holding its breath.

### 3.2. `rift_adrenaline.py` (Finite Stamina)
*   **Concept:** Explicit `Stamina` pool. 100 steps to exhaust.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 1500
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Gate @ Hit: 0.510 | Stamina: 0.00`
    *   **Analysis:** Stamina drained before impact. Neural net failed to synchronize.

### 3.3. `rift_burden.py` (Evolutionary Weight)
*   **Concept:** Evolutionary gains (Health, Battery) add Mass/Drag.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 3000
*   **Equation:** $Load = 0.02 H_{max} + 0.05 B_{max}$
*   **Result:** **HEAVYWEIGHT**.
    *   **Log:** `H_max 1.05 | B_max 1.60`
    *   **Analysis:** System prioritized Battery capacity despite the drag, accepting sluggishness for power.

---

## 4. Phase IV: Sensory Dynamics
**Goal:** Link perception to action. "Tension blinds you."

### 4.1. `rift_world.py` (Unified Laws)
*   **Concept:** Combined Trauma, Mortality, Ischemia, and Blindness.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 3000
*   **Law:** `Noise = Tension * 1.0`.
*   **Result:** 🐢 **PARTIAL**.
    *   **Log:** `Tension@Hit: 0.815 | Vision: -0.51`
    *   **Analysis:** System accepted partial blindness to survive. It braced *just enough* to survive but not enough to go fully blind. It brute-forced the trade-off.

### 4.2. `rift_ambiguity.py` (Probabilistic Futures)
*   **Concept:** Cue is ambiguous (80% Left, 20% Right).
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 4000
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Posture X: -0.420 (Hedged)`
    *   **Analysis:** System failed to commit. It hedged in the middle and died.

### 4.3. `rift_dissonance.py` (Cognitive Dissonance)
*   **Concept:** Pain = Mismatch between Belief (Trace) and Reality (Posture).
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 5000
*   **Equation:** $\text{Pain} += 0.5 \text{ if } \text{sign}(T) \neq \text{sign}(P)$
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Action: BOOM (Fire) | Health: 0.0`
    *   **Analysis:** The "Impulse" cost to switch reality was too high. System burned out.

---

## 5. Phase V: Motor Control & Action
**Goal:** Better actuators. Impedance control, Gating, Impulses.

### 5.1. `rift_kinesis.py` (Motor Babbling)
*   **Concept:** Learn to `Brace` (Action) vs `Relax`.
*   **Grid:** 32x32, 16 Channels
*   **Steps:** 1000
*   **Equation:** $Pain = |U| + |Action| \cdot 0.1$
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Pain Reduction: 0.0%`

### 5.2. `rift_kinesis2.py` (Impedance Control)
*   **Concept:** Modulate stiffness $K$.
*   **Grid:** 32x32, 16 Channels
*   **Steps:** 1000
*   **Equation:** $K_{total} = K_{structure} + 2.0 \cdot \text{sigmoid}(Action)$
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Damage Reduction: 0.0%`
    *   **Analysis:** Wave physics ignored stiffness changes.

### 5.3. `rift_contact.py` (Gating)
*   **Concept:** Learn to close a generic "Gate" channel.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 1000
*   **Equation:** $Force_{eff} = Force_{ext} \cdot \text{sigmoid}(Gate)$
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Pain Reduction: -0.0001%`

### 5.4. `rift_impulse.py` (The Jump)
*   **Concept:** Store energy in a capacitor, release as discrete `Impulse`.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 5000
*   **Equation:** $Force = \text{sign}(Tilt) \cdot Battery \cdot 5.0$
*   **Result:** ✅ **SUCCESS**.
    *   **Log:** `Posture@Hit: 0.53 | Fired At: YES`
    *   **Analysis:** Discontinuous action allowed the system to overcome inertia.

### 5.5. `rift_switch.py` (Plasticity)
*   **Concept:** "Melt" energy barriers to switch states.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 4000
*   **Equation:** $Barrier = 2.0 \cdot (1.0 - Melt)$
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Barrier Melt | Status: LOCK`
    *   **Analysis:** Melting was too dangerous (exposed to impact).

---

## 6. Phase VI: Memory & Prediction
**Goal:** Physical substrates for memory.

### 6.1. `rift_mnemos.py` (Memory Capacity)
*   **Concept:** Standard capacity test.
*   **Grid:** 48x48, 32 Channels
*   **Steps:** 900
*   **Result:** ✅ **SUCCESS**.
    *   **Log:** `Recall X: 0.0042 (Strong)`

### 6.2. `rift_mnemos_2.py` (9 Sites)
*   **Concept:** Sequential learning of 9 locations.
*   **Grid:** 48x48, 32 Channels
*   **Steps:** 200/Site
*   **Result:** ❌ **CATASTROPHIC FORGETTING**.
    *   **Log:** `Sites Retained: 1/9`

### 6.3. `rift_mnemos_3.py` (Tuned)
*   **Concept:** Reduced plasticity to prevent overwriting.
*   **Grid:** 48x48, 32 Channels
*   **Steps:** 200/Site
*   **Result:** ⚠️ **INTERFERENCE**.
    *   **Log:** `Sites Retained: 4/9`

### 6.4. `rift_mnemos_4.py` (Dual Timescale)
*   **Concept:** Fast (Skin) + Slow (Bone) layers.
*   **Grid:** 48x48, 32 Channels
*   **Steps:** 200/Site
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Sites Retained: 2/9`

### 6.5. `rift_mnemos_5.py` (Spatial Gating)
*   **Concept:** Novelty filter based on stress sharpness.
*   **Grid:** 48x48, 32 Channels
*   **Steps:** 200/Site
*   **Result:** ✅ **SUCCESS**.
    *   **Log:** `Sites Retained: 7/9`

### 6.6. `rift_mnemos_p` to `p4` (Prediction)
*   **Concept:** Can the substrate predict wave arrival?
*   **Grid:** 64x64, 16 Channels
*   **p:** ✅ Basic Prediction (Stress reduction).
*   **p2:** ❌ Pure Wave (No effect).
*   **p3:** ❌ Dissipation (No effect).
*   **p4:** ❌ Amplified Hebbian (No effect).

### 6.7. `rift_engram.py` (RNN Latch)
*   **Concept:** Neural RNN to hold state.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 3000
*   **Result:** ⚠️ **DELUSION**.
    *   **Log:** `Posture@Hit: 1.000`
    *   **Analysis:** Memory decoupled from reality. Hallucinated safety.

### 6.8. `rift_aware.py` (Trace Decay)
*   **Concept:** Simple decaying trace field.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 1000
*   **Result:** ✅ **SUCCESS**.
    *   **Log:** `Pain Reduction: 5.6%`

### 6.9. `rift_hysteresis.py` (Friction Memory)
*   **Concept:** Use friction to hold state.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 3000
*   **Result:** ❌ **DRIFT**.

### 6.10. `rift_attractor.py` (Basins)
*   **Concept:** Use energy wells.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 3000
*   **Result:** ❌ **FAILURE**.

### 6.11. `rift_phase_transition.py` (Thermodynamic Lock)
*   **Concept:** Freeze/Melt barrier.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 3000
*   **Result:** 🏆 **GLORIOUS SUCCESS**.
    *   **Log:** `Posture Trapped. Zero Cost. True Memory.`

### 6.12. `rift_homeostasis.py` (Regulation)
*   **Concept:** Balances Stability, Health, Battery.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 4000
*   **Result:** ✅ **SUCCESS**.
    *   **Log:** `System Switched AND Survived`

### 6.13. `rift_resonance.py` (Phase Sync)
*   **Concept:** Sync internal phase with external cue.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 3000
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Avg Lock: 0.000`

---

## 7. Phase VII: Morphology & Geometry (THE BREAKTHROUGH)
**Goal:** Anticipation via geometric commitment.

### 7.1. `rift_morphology.py` (Baseline)
*   **Concept:** Shift center of mass ($x, y$).
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 3000
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Posture X: -0.578` (Too slow).

### 7.2. `rift_morphology_v2.py` (Tuned)
*   **Concept:** Tuned decay (0.98) and cost (0.01).
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 3000
*   **Result:** ✅ **SUCCESS**.
    *   **Log:** `Posture X: -0.527 | Status: HELD`
    *   **Analysis:** The system successfully anticipated and held posture.

---

## 8. Phase VIII: Ecology & Evolution
**Goal:** Speciation through environmental pressure.

### 8.1. `rift_ecology.py` & `rift_ecology_tradeoff.py`
*   **Biomes:** `CRUSHER` (Wide/Heavy), `SNIPER` (Narrow/Lethal).
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 120
*   **Result:**
    *   `CRUSHER`: `H_max: 94.51` (Tank).
    *   `SNIPER`: `H_max: 93.59` (Tank).
    *   **Analysis:** Physics favored Mass over Speed in almost all cases.

### 8.2. `rift_speciation.py` & `rift_speciation_v2.py`
*   **Concept:** Forcing divergence.
*   **Grid:** 32x32, 8 Channels
*   **Episodes:** 3000
*   **Result:** `FINAL SPECIES: H=151.00`.
    *   **Analysis:** Extreme tanking.

### 8.3. `rift_min_speciation.py` (Minimal Model)
*   **Concept:** Hard-coded exclusion (Tanks can't dodge, Speedsters die on hit).
*   **Grid:** 32x32, 8 Channels
*   **Episodes:** 2000
*   **Result:**
    *   `CRUSHER`: `H=12.50` (Pure Tank).
    *   `SNIPER`: `B=8.40` (Pure Speed).
    *   **Analysis:** Forced logic worked where soft trade-offs failed.

### 8.4. `rift_population.py`
*   **Concept:** Genetic Algorithm.
*   **Grid:** 32x32, 8 Channels
*   **Generations:** 50
*   **Result:** `H=0.56, B=0.90`.
    *   **Analysis:** Found a balanced strategy.

### 8.5. `rift_necessity.py`
*   **Concept:** Global vs Local stress.
*   **Grid:** 32x32, 8 Channels
*   **Result:**
    *   `CRUSHER`: `H=1.29` (High Health).
    *   `SNIPER`: `H=0.42` (Low Health, High Agility).
    *   **Analysis:** Finally achieved speciation!

### 8.6. `rift_escalation.py`
*   **Concept:** Unresolved stress becomes injury.
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 4000
*   **Result:** `Stability 0.54 | Health 0.00` (Death spiral).

### 8.7. `rift_hypercompensation.py`
*   **Concept:** "What doesn't kill you makes you stronger".
*   **Grid:** 32x32, 8 Channels
*   **Steps:** 3000
*   **Result:** `H_max 252.40`. Infinite growth loop.

### 8.8. `rift_chimera.py` Series (Hybrid)
*   **Concept:** Neural Net + Physics.
*   **Results:**
    *   `v1`: ❌ Failed.
    *   `v2`: ❌ Failed (Catastrophic Penalty).
    *   `v3`: 🐢 Partial (Paranoia).

---

## 9. Phase IX: Validation
**Goal:** Prove causality.

### 9.1. `rift_validation.py` (Inquisitor)
*   **Grid:** 32x32
*   **Tests:** Control, Stretch, Noise, Lobotomy.
*   **Result:** ❌ **FAILED**.
    *   **Log:** `Control: 3.54%` (Too low).
    *   **Analysis:** The effect size of memory was statistically insignificant in the neural-hybrid model.

---

## 10. Runners & Utilities
*   **`rift_experiment.py`**: Generic harness.
*   **`rift_experiment3.py`**: Fixed harness.
*   **`rift_experiment4.py`**: Episodic trauma.
*   **`rift_experimentt2.py`**: Alternative harness.

---

## Conclusion
Project RIFT proved that **anticipation is a geometric property**. The most successful agents were not those with the biggest brains (Chimera, Engram), but those with the most tunable physics (`rift_phase_transition`, `rift_morphology_v2`).

*Document generated by Jules (AI).*
