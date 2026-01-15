(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\babel_lesion.py
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🔪 SURGICAL PROTOCOL INITIATED ON cuda
✅ LOADING PATIENT: chimera_babel_dna.pt
D:\aria\aria_ai\aria_ai_assistant\AGI\babel_lesion.py:62: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  organism.load_state_dict(torch.load(SAVE_FILE))

🔎 RUNNING BASELINE (INTACT BRAIN)...
   Baseline Geography Fitness: 0.0189

🔪 PERFORMING LOBOTOMY (REMOVING TOP-LEFT QUADRANT)...
   Post-Surgery Geography Fitness: -0.0142

📋 DIAGNOSTIC REPORT:
   Performance Drop: 0.0331
   ✅ SUCCESS: INDEPENDENT MODULES CONFIRMED.
      The Geography Organ survived the death of the Royalty Organ.
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 


======================================


(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\weaver_baseline.py                               
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🧶 PROJECT WEAVER: BASELINE PROTOCOL ON cuda
✅ LOADING PATIENT: chimera_babel_dna.pt
D:\aria\aria_ai\aria_ai_assistant\AGI\weaver_baseline.py:65: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  organism.load_state_dict(torch.load(SAVE_FILE))

🧩 TEST: JOINT REASONING (KING + PARIS)
   Center Binding Fitness: 0.0897
   ✅ EXPECTED FAILURE: The concepts remained isolated.
      (This confirms we need a Thalamic Bridge)
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 

====================================


(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\thalamus.py
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🌡️ PROJECT WEAVER: THALAMIC ATTENTION LOOP ON cuda
✅ LOADING PATIENT: chimera_babel_dna.pt
D:\aria\aria_ai\aria_ai_assistant\AGI\thalamus.py:63: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  organism.load_state_dict(torch.load(SAVE_FILE))

🧠 STARTING COGNITIVE CYCLE (King + Paris)...
STEP   | TEMP   | DECAY  | BINDING (FITNESS)  | STATUS
-----------------------------------------------------------------
0      | 0.01   | 0.920  | -0.0443             | 🔥 Focusing...
2      | 0.03   | 0.921  | 0.0120             | 🔥 Focusing...
4      | 0.05   | 0.923  | 0.0385             | 🔥 Focusing...
6      | 0.07   | 0.924  | 0.0794             | 🔥 Focusing...
8      | 0.09   | 0.926  | 0.1069             | 🔥 Focusing...
10     | 0.11   | 0.927  | 0.1259             | 🔥 Focusing...
12     | 0.13   | 0.928  | 0.1414             | 🔥 Focusing...
14     | 0.15   | 0.930  | 0.1618             | 🔥 Focusing...
16     | 0.17   | 0.931  | 0.1701             | 🔥 Focusing...
18     | 0.19   | 0.933  | 0.1835             | 🔥 Focusing...
20     | 0.21   | 0.934  | 0.1944             | 🔥 Focusing...
22     | 0.23   | 0.935  | 0.2005             | 🔥 Focusing...
24     | 0.25   | 0.937  | 0.2060             | 🔥 Focusing...
26     | 0.27   | 0.938  | 0.2115             | 🔥 Focusing...
28     | 0.29   | 0.940  | 0.2142             | 🔥 Focusing...
30     | 0.31   | 0.941  | 0.2172             | 🔥 Focusing...
32     | 0.33   | 0.942  | 0.2178             | 🔥 Focusing...
34     | 0.35   | 0.944  | 0.2129             | 🔥 Focusing...
36     | 0.37   | 0.945  | 0.2092             | 🔥 Focusing...
38     | 0.39   | 0.947  | 0.2055             | 🔥 Focusing...
40     | 0.41   | 0.948  | 0.2021             | 🔥 Focusing...
42     | 0.43   | 0.949  | 0.1970             | 🔥 Focusing...
44     | 0.45   | 0.951  | 0.1865             | 🔥 Focusing...
46     | 0.47   | 0.952  | 0.1826             | 🔥 Focusing...
48     | 0.49   | 0.954  | 0.1722             | 🔥 Focusing...
50     | 0.51   | 0.955  | 0.1624             | 🔥 Focusing...
52     | 0.53   | 0.956  | 0.1527             | 🔥 Focusing...
54     | 0.55   | 0.958  | 0.1485             | 🔥 Focusing...
56     | 0.57   | 0.959  | 0.1443             | 🔥 Focusing...
58     | 0.59   | 0.961  | 0.1412             | 🔥 Focusing...
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 


=================================



.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\weaver_loom.py    
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🧶 PROJECT WEAVER: THE LOOM (SCAFFOLDING) ON cuda
✅ LOADING PATIENT: chimera_babel_dna.pt
D:\aria\aria_ai\aria_ai_assistant\AGI\weaver_loom.py:75: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  organism.load_state_dict(torch.load(SAVE_FILE))

🧠 ACTIVATING WORKING MEMORY TEMPLATE...
STEP   | TEMP   | DECAY  | BINDING    | STATUS
------------------------------------------------------------
0      | 0.02   | 0.920  | -0.1381     | 🔥 Building Loom
5      | 0.12   | 0.926  | -0.1169     | 🔥 Building Loom
10     | 0.22   | 0.932  | -0.1174     | 🔥 Building Loom
15     | 0.32   | 0.938  | -0.1186     | 🔥 Building Loom
20     | 0.42   | 0.944  | 0.0744     | 🔥 Building Loom
25     | 0.52   | 0.950  | 0.0527     | 🔥 Building Loom
30     | 0.62   | 0.956  | 0.0547     | 🔥 Building Loom
35     | 0.72   | 0.962  | 0.0672     | 🔥 Building Loom
40     | 0.82   | 0.968  | 0.0680     | 🔥 Building Loom
45     | 0.92   | 0.974  | 0.0494     | 🔥 Building Loom
50     | 1.00   | 0.980  | 0.0401     | 🔥 Building Loom
55     | 1.00   | 0.980  | 0.0237     | 🔥 Building Loom
60     | 1.00   | 0.980  | 0.0475     | 🔥 Building Loom
65     | 1.00   | 0.980  | 0.0460     | 🔥 Building Loom
70     | 1.00   | 0.980  | 0.0384     | 🔥 Building Loom
75     | 1.00   | 0.980  | 0.0814     | 🔥 Building Loom
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 




==================================================


(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\project_weaver_integrated.py
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🧬 PROJECT WEAVER: INTEGRATED HANDSHAKE PROTOCOL ON cuda
📚 Initializing Semantic Projector...

🏗️ PHASE 1: GROWING SPLIT BRAIN (Fast Track)...
Gen 0: Royal 0.47 | Geo 0.41
Gen 50: Royal 0.28 | Geo 0.30
Gen 100: Royal 0.44 | Geo 0.55
Gen 150: Royal 0.15 | Geo 0.28
🏆 BABEL READY. Best Fitness: 0.55

🤝 PHASE 2: ACTIVATING BINDING SITE...
STEP   | CENTER E   | BINDING    | STATUS
--------------------------------------------------
0      | 0.0000     | 0.0000     | ☠️ Toxic (Decay)
5      | 0.0000     | 0.0000     | ☠️ Toxic (Decay)
10     | 0.0000     | -0.0030     | ☠️ Toxic (Decay)
15     | 0.0000     | 0.0335     | ☠️ Toxic (Decay)
20     | 0.0000     | 0.0357     | ☠️ Toxic (Decay)
25     | 0.0000     | 0.0974     | ☠️ Toxic (Decay)
30     | 0.0000     | 0.1132     | ☠️ Toxic (Decay)
35     | 0.0000     | 0.1080     | ☠️ Toxic (Decay)
40     | 0.0000     | 0.1406     | ☠️ Toxic (Decay)
45     | 0.0000     | 0.1496     | ☠️ Toxic (Decay)
50     | 0.0000     | 0.1673     | ☠️ Toxic (Decay)
55     | 0.0000     | 0.1794     | ☠️ Toxic (Decay)
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 


==============================================================


(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\weaver_pressure.py
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🌋 PROJECT WEAVER: PRESSURE COOKER PROTOCOL ON cuda
📚 Initializing Semantic Projector...

🏗️ PHASE 1: GROWING SPLIT BRAIN...
Gen 0: Royal 0.48 | Geo 0.55
Gen 50: Royal 0.39 | Geo 0.53
Gen 100: Royal 0.39 | Geo 0.43
Gen 150: Royal 0.19 | Geo 0.27
Gen 200: Royal 0.44 | Geo 0.35
🏆 BABEL READY. Fitness: 0.65

🔥 PHASE 2: APPLYING HOMEOSTATIC PRESSURE...
STEP   | STRESS | ISLAND DECAY | CENTER E   | BINDING    | STATUS
---------------------------------------------------------------------------
0      | 0.005  | 0.915        | 0.0000     | 0.0000     | 🔥 Squeezing...
5      | 0.030  | 0.890        | 0.0000     | 0.0000     | 🔥 Squeezing...
10     | 0.055  | 0.865        | 0.0000     | 0.1239     | 🔥 Squeezing...
15     | 0.080  | 0.840        | 0.0000     | 0.0911     | 🔥 Squeezing...
20     | 0.105  | 0.815        | 0.0000     | 0.2137     | 🔥 Squeezing...
25     | 0.130  | 0.790        | 0.0000     | 0.2556     | 🔥 Squeezing...
30     | 0.150  | 0.770        | 0.0000     | 0.1745     | 🔥 Squeezing...
35     | 0.150  | 0.770        | 0.0000     | 0.2359     | 🔥 Squeezing...
40     | 0.150  | 0.770        | 0.0000     | 0.2853     | 🔥 Squeezing...
45     | 0.150  | 0.770        | 0.0000     | 0.3099     | 🔥 Squeezing...
50     | 0.150  | 0.770        | 0.0000     | 0.2406     | 🔥 Squeezing...
55     | 0.150  | 0.770        | 0.0000     | 0.2939     | 🔥 Squeezing...
60     | 0.150  | 0.770        | 0.0000     | 0.2551     | 🔥 Squeezing...
65     | 0.150  | 0.770        | 0.0000     | 0.2592     | 🔥 Squeezing...
70     | 0.150  | 0.770        | 0.0000     | 0.2578     | 🔥 Squeezing...
75     | 0.150  | 0.770        | 0.0000     | 0.1838     | 🔥 Squeezing...
80     | 0.150  | 0.770        | 0.0000     | 0.1218     | 🔥 Squeezing...
85     | 0.150  | 0.770        | 0.0000     | 0.0598     | 🔥 Squeezing...
90     | 0.150  | 0.770        | 0.0000     | 0.0261     | 🔥 Squeezing...
95     | 0.150  | 0.770        | 0.0000     | 0.0125     | 🔥 Squeezing...
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 



=====================================================================


PS D:\aria\aria_ai\aria_ai_assistant> & D:\aria\aria_ai\aria_ai_assistant\.venv\Scripts\Activate.ps1
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\experiment_scale.py                      
🌍 LAUNCHING 120x120 PLANETARY SIMULATION ON cuda...
⏳ Evolving Physics...
   Step 0/100 complete.
   Step 10/100 complete.
   Step 20/100 complete.
   Step 30/100 complete.
   Step 40/100 complete.
   Step 50/100 complete.
   Step 60/100 complete.
   Step 70/100 complete.
   Step 80/100 complete.
   Step 90/100 complete.
📸 Rendering Satellite Imagery...
✅ Saved to bmoi_scale_120.png
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\experiment_benchmark.py
🧪 STARTING RIGOROUS BENCHMARK SUITE (v0.2)...

🌱 RUNNING SEED 42...
Traceback (most recent call last):
  File "D:\aria\aria_ai\aria_ai_assistant\AGI\experiment_benchmark.py", line 31, in <module>
    child = trainer.mutate(organism, rate=0.02)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\aria\aria_ai\aria_ai_assistant\AGI\bmoi_core.py", line 113, in mutate
    child = copy.deepcopy(organism)
            ^^^^
NameError: name 'copy' is not defined
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> pip install copy
ERROR: Could not find a version that satisfies the requirement copy (from versions: none)
ERROR: No matching distribution found for copy

[notice] A new release of pip is available: 24.0 -> 25.3
[notice] To update, run: python.exe -m pip install --upgrade pip
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\experiment_benchmark.py
🧪 STARTING RIGOROUS BENCHMARK SUITE (v0.2)...

🌱 RUNNING SEED 42...

🌱 RUNNING SEED 101...

🌱 RUNNING SEED 202...

🌱 RUNNING SEED 303...

🌱 RUNNING SEED 555...

📊 BENCHMARK COMPLETE.

📋 FINAL STATISTICS:
                      mean  std
type
Genesis_Baseline  0.767029  0.0
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\experiment_benchmark.py
🧪 STARTING RIGOROUS BENCHMARK SUITE (v0.2.1 - Fix)...

🌱 RUNNING SEED 42...

🌱 RUNNING SEED 101...

🌱 RUNNING SEED 202...

🌱 RUNNING SEED 303...

🌱 RUNNING SEED 555...

📊 BENCHMARK COMPLETE.
Traceback (most recent call last):
  File "D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\pandas\core\indexes\base.py", line 3812, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7096, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 400

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\aria\aria_ai\aria_ai_assistant\AGI\experiment_benchmark.py", line 62, in <module>
    final_stats = df[df["generation"] == (GENS - (GENS % 100)) if GENS % 100 != 0 else GENS-100]
                  ~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\pandas\core\frame.py", line 4107, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\pandas\core\indexes\base.py", line 3819, in get_loc
    raise KeyError(key) from err
KeyError: 400
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\experiment_spartan_v2.py
🛡️ LAUNCHING SPARTAN PROTOCOL V2 (Comparative Stress Test)...

👶 Training Naive Organism (No Damage)...
   Naive Training Complete. Best Fitness: 0.814

⚔️ Training Spartan Organism (20% Chronic Damage)...
   Spartan Training Complete. Best Fitness: 0.882

📉 RUNNING DAMAGE CURVES...
   Damage 0%: Naive=0.813 | Spartan=0.603
   Damage 10%: Naive=0.709 | Spartan=0.803
   Damage 30%: Naive=0.462 | Spartan=0.884
   Damage 50%: Naive=0.381 | Spartan=0.836
   Damage 70%: Naive=0.346 | Spartan=0.795
   Damage 90%: Naive=0.330 | Spartan=0.767

✅ Data saved to bmoi_spartan_results.csv
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\bmoi_spartan_curve.py
✅ Chart generated: bmoi_spartan_curve.png
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\weaver_logic.py   
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🧠 PROJECT WEAVER: CONTEXTUAL LOGIC GATE ON cuda
📚 Initializing Semantic Projector...

🏫 TRAINING: DUAL-CONTEXT CURRICULUM...
Traceback (most recent call last):
  File "D:\aria\aria_ai\aria_ai_assistant\AGI\weaver_logic.py", line 100, in <module>
    grid_a[:, :, 4, 4] = v_summer.view(1, HIDDEN_DIM, 1, 1)
    ~~~~~~^^^^^^^^^^^^
RuntimeError: expand(torch.cuda.FloatTensor{[64, 1, 1]}, size=[1, 64]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (3)
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\weaver_logic.py
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🧠 PROJECT WEAVER: CONTEXTUAL LOGIC GATE (v2 FIXED) ON cuda
📚 Initializing Semantic Projector...

🏫 TRAINING: DUAL-CONTEXT CURRICULUM...
Gen 0: Summer->Beach: -0.11 | Winter->Sauna: -0.05 | Avg: -0.080
Gen 50: Summer->Beach: 0.34 | Winter->Sauna: 0.31 | Avg: 0.325
Gen 100: Summer->Beach: 0.28 | Winter->Sauna: 0.21 | Avg: 0.245
Gen 150: Summer->Beach: 0.48 | Winter->Sauna: 0.45 | Avg: 0.466
Gen 200: Summer->Beach: 0.62 | Winter->Sauna: 0.56 | Avg: 0.594
Gen 250: Summer->Beach: 0.60 | Winter->Sauna: 0.61 | Avg: 0.605
Gen 300: Summer->Beach: 0.51 | Winter->Sauna: 0.53 | Avg: 0.520
Gen 350: Summer->Beach: 0.69 | Winter->Sauna: 0.69 | Avg: 0.688
🏆 LOGIC GATE TRAINED. Best Fitness: 0.707

🕵️ RUNNING INFERENCE TEST...
Can the organism switch outputs based on context?
Context: SUMMER + HOT -> Beach (0.51) vs Sauna (0.50)
Context: WINTER + HOT -> Beach (0.49) vs Sauna (0.48)
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\weaver_sniper.py
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🎯 PROJECT WEAVER: TEMPORAL SNIPER PROTOCOL ON cuda
📚 Initializing Semantic Projector...

🏫 TRAINING: SURVIVING THE SNIPER...
Gen 0: Summer->Beach: 0.00 | Winter->Sauna: 0.00 | Avg: 0.000
Gen 50: Summer->Beach: 0.00 | Winter->Sauna: 0.00 | Avg: 0.000
Gen 100: Summer->Beach: 0.00 | Winter->Sauna: 0.00 | Avg: 0.000
Gen 150: Summer->Beach: 0.00 | Winter->Sauna: 0.00 | Avg: 0.000
Gen 200: Summer->Beach: 0.00 | Winter->Sauna: 0.00 | Avg: 0.000
Gen 250: Summer->Beach: 0.00 | Winter->Sauna: 0.00 | Avg: 0.000
Gen 300: Summer->Beach: 0.00 | Winter->Sauna: 0.00 | Avg: 0.000
Gen 350: Summer->Beach: 0.00 | Winter->Sauna: 0.00 | Avg: 0.000
Gen 400: Summer->Beach: 0.00 | Winter->Sauna: 0.00 | Avg: 0.000
Gen 450: Summer->Beach: 0.00 | Winter->Sauna: 0.00 | Avg: 0.000
🏆 SNIPER SURVIVOR TRAINED. Best Fitness: 0.000

🕵️ CHECKING FOR PURPLE MUD...
(If one score is high and the other low, we solved it. If both are 0.5, we failed.)
Context: SUMMER -> Beach: 0.00 | Sauna: 0.00
Context: WINTER -> Beach: 0.00 | Sauna: 0.00

⚠️ FAILURE: Still Muddled.
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\weaver_sniper.py
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🎯 PROJECT WEAVER: PHASED SNIPER PROTOCOL ON cuda
📚 Initializing Semantic Projector...

🏫 TRAINING: EXPLORE -> COMPETE -> COMMIT...
Traceback (most recent call last):
  File "D:\aria\aria_ai\aria_ai_assistant\AGI\weaver_sniper.py", line 123, in <module>
    grid_a, _ = mutant(grid_a, prev_a, sniper_active=sniper_on)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\aria\aria_ai\aria_ai_assistant\AGI\weaver_sniper.py", line 82, in forward
    return new_grid, delta if prev_grid is not None else 0.0
                     ^^^^^
UnboundLocalError: cannot access local variable 'delta' where it is not associated with a value
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 


(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\weaver_sniper.py
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🎯 PROJECT WEAVER: PHASED SNIPER PROTOCOL (FIXED) ON cuda
📚 Initializing Semantic Projector...

🏫 TRAINING: EXPLORE -> COMPETE -> COMMIT...
Gen 0: Summer->Beach: -0.04 | Winter->Sauna: -0.09 | Avg: -0.064
Gen 50: Summer->Beach: 0.25 | Winter->Sauna: 0.30 | Avg: 0.273
Gen 100: Summer->Beach: 0.26 | Winter->Sauna: 0.28 | Avg: 0.270
Gen 150: Summer->Beach: 0.41 | Winter->Sauna: 0.44 | Avg: 0.426
Gen 200: Summer->Beach: 0.54 | Winter->Sauna: 0.53 | Avg: 0.537
Gen 250: Summer->Beach: 0.41 | Winter->Sauna: 0.37 | Avg: 0.392
Gen 300: Summer->Beach: 0.42 | Winter->Sauna: 0.40 | Avg: 0.413
Gen 350: Summer->Beach: 0.57 | Winter->Sauna: 0.59 | Avg: 0.580
Gen 400: Summer->Beach: 0.54 | Winter->Sauna: 0.53 | Avg: 0.533
Gen 450: Summer->Beach: 0.49 | Winter->Sauna: 0.48 | Avg: 0.482
🏆 PHASED LOGIC TRAINED. Best Fitness: 0.678

🕵️ FINAL DECISION CHECK...
Context: SUMMER -> Beach: 0.67 | Sauna: 0.68
Context: WINTER -> Beach: 0.68 | Sauna: 0.69

⚠️ FAILURE: Indecisive. (Gap: -0.01, 0.00)
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\weaver_contrastive.py
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
⚖️ PROJECT WEAVER: CONTRASTIVE LOGIC PROTOCOL ON cuda
📚 Initializing Semantic Projector...

🏫 TRAINING: CONTRASTIVE CURRICULUM (Reward Target, Punish Distractor)...
Gen 0:
  Summer: Beach=-0.05 | Sauna=-0.07 (Gap: 0.02)
  Winter: Sauna=-0.02 | Beach=-0.01 (Gap: -0.01)
Gen 50:
  Summer: Beach=0.33 | Sauna=0.31 (Gap: 0.02)
  Winter: Sauna=0.33 | Beach=0.35 (Gap: -0.02)
Gen 100:
  Summer: Beach=0.37 | Sauna=0.35 (Gap: 0.01)
  Winter: Sauna=0.34 | Beach=0.35 (Gap: -0.01)
Gen 150:
  Summer: Beach=0.13 | Sauna=0.14 (Gap: -0.01)
  Winter: Sauna=0.16 | Beach=0.15 (Gap: 0.01)
Gen 200:
  Summer: Beach=0.10 | Sauna=0.08 (Gap: 0.01)
  Winter: Sauna=0.08 | Beach=0.09 (Gap: -0.01)
Gen 250:
  Summer: Beach=0.32 | Sauna=0.31 (Gap: 0.01)
  Winter: Sauna=0.33 | Beach=0.35 (Gap: -0.02)
Gen 300:
  Summer: Beach=0.33 | Sauna=0.34 (Gap: -0.01)
  Winter: Sauna=0.33 | Beach=0.33 (Gap: 0.00)
Gen 350:
  Summer: Beach=-0.08 | Sauna=-0.08 (Gap: -0.01)
  Winter: Sauna=-0.08 | Beach=-0.09 (Gap: 0.01)
Gen 400:
  Summer: Beach=0.24 | Sauna=0.23 (Gap: 0.01)
  Winter: Sauna=0.21 | Beach=0.21 (Gap: -0.01)
Gen 450:
  Summer: Beach=0.41 | Sauna=0.41 (Gap: 0.00)
  Winter: Sauna=0.43 | Beach=0.44 (Gap: -0.00)
Gen 500:
  Summer: Beach=0.03 | Sauna=0.03 (Gap: 0.01)
  Winter: Sauna=0.10 | Beach=0.11 (Gap: -0.01)
Gen 550:
  Summer: Beach=0.38 | Sauna=0.35 (Gap: 0.03)
  Winter: Sauna=0.36 | Beach=0.38 (Gap: -0.03)
🏆 CONTRASTIVE LOGIC TRAINED. Best Fitness: 0.249

🕵️ FINAL DECISION GAP CHECK...
Context: SUMMER -> Beach: 0.51 | Sauna: 0.49 | GAP: 0.02
Context: WINTER -> Sauna: 0.48 | Beach: 0.50 | GAP: -0.02
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\AGI\weaver_polarity.py   
D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\transformers\utils\hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
🧲 PROJECT WEAVER: POLARITY PROTOCOL (Antipodal Targets) ON cuda
📚 Initializing Semantic Projector...

🏫 TRAINING: POLARITY CURRICULUM...
Gen 0: Summer->Beach: 0.10 | Winter->Sauna: -0.13 | Avg: -0.018
Gen 50: Summer->Beach: 0.18 | Winter->Sauna: -0.07 | Avg: 0.055
Gen 100: Summer->Beach: 0.15 | Winter->Sauna: -0.07 | Avg: 0.041
Gen 150: Summer->Beach: 0.26 | Winter->Sauna: -0.18 | Avg: 0.039
Gen 200: Summer->Beach: 0.02 | Winter->Sauna: 0.11 | Avg: 0.062
Gen 250: Summer->Beach: -0.09 | Winter->Sauna: 0.19 | Avg: 0.049
Gen 300: Summer->Beach: 0.01 | Winter->Sauna: 0.08 | Avg: 0.042
Gen 350: Summer->Beach: -0.06 | Winter->Sauna: 0.21 | Avg: 0.077
Gen 400: Summer->Beach: 0.15 | Winter->Sauna: -0.08 | Avg: 0.035
Gen 450: Summer->Beach: -0.21 | Winter->Sauna: 0.29 | Avg: 0.037
Gen 500: Summer->Beach: 0.05 | Winter->Sauna: 0.14 | Avg: 0.094
Gen 550: Summer->Beach: -0.05 | Winter->Sauna: 0.25 | Avg: 0.097
🏆 POLARITY LOGIC TRAINED. Best Fitness: 0.108

🕵️ CHECKING THE POLARITY GAP...
Context: SUMMER -> Beach: -0.07 | Sauna: 0.07
Context: WINTER -> Sauna: 0.29 | Beach: -0.29

⚠️ FAILURE: Still Muddled. (Total Gap: 0.43)
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 

