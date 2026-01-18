# PROJECT RIFT: THE COMPLETE CHRONICLE
**From Semantic Tectonics to Irreversible Commitment (Phases I - X)**

This document serves as the master log for the entire RIFT research trajectory, organizing every experiment, hypothesis, and result into a coherent narrative of failure and discovery.

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

---

## 2. Phase II: Mortality & Perception
**Goal:** Introduce consequences (Mortality) and limits (Blindness) to force the agent to value its own existence.

### 2.1. `rift_scare.py` (Mortal Body)
*   **Concept:** Continuous gradients + Mortality pressure forces agency.
*   **Physics:** Health tracks accumulated damage. If Health <= 0, Episode Ends (Reward = -1).
*   **Result:** ✅ **SUCCESS**.
    *   **Log:** `Episode 1800 | Pain: 7.06 | Tension: 1.000 | Health: 1.00`
    *   **Analysis:** The agent learned to clamp tension to 1.0 continuously ("Paranoia"). It survived, but was inefficient.

### 2.2. `rift_world.py` (Unified Pressure)
*   **Concept:** "Paranoia causes Blindness". High tension increases sensory noise.
*   **Physics:** Observation Noise $\propto$ Tension.
*   **Result:** 🐢 **PARTIAL**.
    *   **Log:** `Survived via Paranoia (Always Tense).`
    *   **Analysis:** The agent chose to be blind and alive rather than relaxed and seeing. It ignored the visual Cue entirely.

---

## 3. Phase III: Constraints (Fatigue & Metabolism)
**Goal:** Force the agent to relax by making tension costly (Fatigue/Metabolism).

### 3.1. `rift_adrenaline.py` (Fatigue)
*   **Concept:** Finite Stamina. Bracing drains a battery.
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Stamina End: 0.00 | Gate @ Hit: 0.510`
    *   **Analysis:** The agent drained its battery before the impact arrived. It could not time the defense.

### 3.2. `rift_chimera_3.py` (Metabolic Constraint)
*   **Concept:** Random timing + High energy cost ($E$) forces reliance on Cue.
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Total Pain: 2.2864 | Gate @ Impact: 1.0000` (but died metabolically).
    *   **Analysis:** The cost of waiting for the cue was too high compared to the probability of impact.

---

## 4. Phase IV: Sensory-Motor Loops
**Goal:** Can the system learn to "gate" or "brace" dynamically?

### 4.1. `rift_kinesis2.py` (Impedance Control)
*   **Concept:** Learn to temporarily stiffen to reject impact energy.
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Damage Reduction: 0.00%`
    *   **Analysis:** The system did not learn the causal link between stiffness and damage reduction.

### 4.2. `rift_aware.py` (Eligibility Traces)
*   **Concept:** A decaying trace field bridges the time gap ($T_{cue} \rightarrow T_{impact}$).
*   **Result:** ✅ **SUCCESS**.
    *   **Log:** `Pain Reduction: 5.6%`
    *   **Analysis:** The trace allowed the system to "predict" across time by carrying the cue forward physically. This was a key insight for later phases.

---

## 5. Phase V: Structural Memory
**Goal:** Move from "Reflex" to "Memory" via permanent deformation.

### 5.1. `rift_mnemos.py` (Trauma Memory)
*   **Concept:** Trauma permanently alters local elasticity (Callus formation).
*   **Result:** ✅ **SUCCESS**.
    *   **Log:** `Initial Trauma X: 0.0237 -> Adapted: 0.0095`
    *   **Analysis:** The system "remembered" the location of impact by hardening that specific region.

### 5.2. `rift_mnemos_5.py` (Spatial Gating)
*   **Concept:** Can the system learn to protect multiple sites without interference?
*   **Result:** ✅ **SUCCESS**.
    *   **Log:** `Sites Retained: 7/9`
    *   **Analysis:** Spatial gating solved the "Catastrophic Forgetting" observed in `mnemos_2`.

---

## 6. Phase VI: Dynamics & Attractors
**Goal:** Use dynamical systems (basins of attraction) to hold decisions.

### 6.1. `rift_attractor.py` (Basins)
*   **Concept:** Decisions persist because gravity holds them in stable basins ($U(x) = x^4 - x^2$).
*   **Result:** ❌ **FAILURE**.
    *   **Log:** `Posture @ Hit: 0.629` (Did not lock).
    *   **Analysis:** The neural controller fought the physics instead of surfing the basin.

### 6.2. `rift_phase_transition.py` (Thermodynamic Lock)
*   **Concept:** Memory is trapped by raising/lowering energy barriers (Annealing).
*   **Result:** 🏆 **GLORIOUS SUCCESS**.
    *   **Log:** `Posture Trapped. Zero Cost. True Memory.`
    *   **Analysis:** This proved that *changing the energy landscape* is more effective than *applying force*.

---

## 7. Phase VII: Morphology & Ecology
**Goal:** Evolution of shape as intelligence.

### 7.1. `rift_morphology_v2.py` (Anticipatory Posture)
*   **Concept:** Slower memory decay prevents premature relaxation.
*   **Result:** ✅ **SUCCESS**.
    *   **Log:** `Body shifted weight and HELD IT.`
    *   **Analysis:** The agent learned to shift its "Center of Mass" left/right based on the Cue, physically encoding the prediction.

### 7.2. `rift_speciation_v2.py` (Divergence)
*   **Concept:** Isolation forces distinct morphological strategies (Crusher vs Sniper).
*   **Result:** 🏁 **FINAL SPECIES:** `H=151.00, B=151.00`
*   **Analysis:** Extreme physiological specialization emerged to handle specific environmental pressures.

---

## 8. Phase X: RIFT-A (Commitment vs Prediction)
**Goal:** The final pivot. Falsify the hypothesis that "Prediction is Intelligence" by proving it fails in irreversible environments.

### 8.1. `rift_a/experiments/diag_trace.py` (Signal Diagnostics)
*   **Calibration:** Initially 0% survival. Tuned Trace Decay $0.8 \rightarrow 0.9$ and Ischemia $0.05 \rightarrow 0.03$.
*   **Result:** Trace visible for 11 steps. Environment verified as satisfiable.

### 8.2. `rift_a/experiments/exp_mini.py` (The Trap)
*   **Concept:** Compare a "Dumb" Geometric Agent vs "Smart" Neural Agents (Transformer, RNN, RL) in a task requiring early irreversible commitment.
*   **Results:**
    *   **Geometric Agent:** **40% Survival**. Commits perfectly at Cue time ($T=45$).
    *   **Transformer:** **0% Survival**. Never commits ($T=100$). Hedges until death.
    *   **RNN:** **0% Survival**. Never commits reliably.
    *   **RL:** **7% Survival**. Panics early ($T=14$) and dies of metabolic cost.
*   **Analysis:** This result effectively falsifies the claim that predictive architectures are universally superior. Under irreversible constraints, prediction leads to fatal hesitation ("Paralysis"), while simple geometric reflexes survive.

---

**Conclusion:**
Project RIFT began by trying to simulate "Semantic Tectonics" (Phase I) and evolved into a rigorous study of **Physical Intelligence**. The journey demonstrated that:
1.  **Memory is Deformation:** (Phase V).
2.  **Agency is Phase Transition:** (Phase VI).
3.  **Prediction is Liability:** (Phase X).

The RIFT-A environment now stands as a benchmark for **Non-Predictive Intelligence**.
