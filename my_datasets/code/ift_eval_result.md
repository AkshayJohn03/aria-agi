(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\my_datasets\code\eval_chkpt3.py
[i] Device: cuda
[i] Loaded SentencePiece tokenizer | vocab=24000 | pad_id=0
[i] Loading val: datasets/processed/test_openorca/val/shard_000
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:42: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.ids = torch.load(ids_path)
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:43: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.lbl = torch.load(lbl_path)
[i] Found 47 checkpoints
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:129: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ck = torch.load(ckpt, map_location="cpu")

=== Evaluating checkpoint_step1049.pt | step=1049 | params=24,985,344 ===
[✓] step=1049 | avg_loss=52.6302 | batches=250 | tokens=304327 | ppl=7.19e+22                                                                                                        

=== Evaluating checkpoint_step1112.pt | step=1112 | params=24,985,344 ===
[✓] step=1112 | avg_loss=52.6302 | batches=250 | tokens=304327 | ppl=7.19e+22                                                                                                        
[skip] checkpoint_step112.pt: checkpoint vocab 60004 != tokenizer vocab 24000

=== Evaluating checkpoint_step1175.pt | step=1175 | params=24,985,344 ===
[✓] step=1175 | avg_loss=51.3283 | batches=250 | tokens=304327 | ppl=1.96e+22                                                                                                        

=== Evaluating checkpoint_step1238.pt | step=1238 | params=24,985,344 ===
[✓] step=1238 | avg_loss=50.2146 | batches=250 | tokens=304327 | ppl=6.43e+21                                                                                                        
[skip] checkpoint_step128.pt: checkpoint vocab 60004 != tokenizer vocab 24000

=== Evaluating checkpoint_step1299.pt | step=1299 | params=24,985,344 ===
[✓] step=1299 | avg_loss=49.0690 | batches=250 | tokens=304327 | ppl=2.04e+21                                                                                                        

=== Evaluating checkpoint_step1362.pt | step=1362 | params=24,985,344 ===
[✓] step=1362 | avg_loss=47.9727 | batches=250 | tokens=304327 | ppl=6.83e+20                                                                                                        

=== Evaluating checkpoint_step1425.pt | step=1425 | params=24,985,344 ===
[✓] step=1425 | avg_loss=47.9727 | batches=250 | tokens=304327 | ppl=6.83e+20                                                                                                        
[skip] checkpoint_step144.pt: checkpoint vocab 60004 != tokenizer vocab 24000

=== Evaluating checkpoint_step1488.pt | step=1488 | params=24,985,344 ===
[✓] step=1488 | avg_loss=46.9537 | batches=250 | tokens=304327 | ppl=2.46e+20                                                                                                        

=== Evaluating checkpoint_step1551.pt | step=1551 | params=24,985,344 ===
[✓] step=1551 | avg_loss=45.8782 | batches=250 | tokens=304327 | ppl=8.41e+19                                                                                                        
[skip] checkpoint_step16.pt: checkpoint vocab 60004 != tokenizer vocab 24000
[skip] checkpoint_step160.pt: checkpoint vocab 60004 != tokenizer vocab 24000

=== Evaluating checkpoint_step1614.pt | step=1614 | params=24,985,344 ===
[✓] step=1614 | avg_loss=44.8821 | batches=250 | tokens=304327 | ppl=3.10e+19                                                                                                        

=== Evaluating checkpoint_step1677.pt | step=1677 | params=24,985,344 ===
[✓] step=1677 | avg_loss=44.8821 | batches=250 | tokens=304327 | ppl=3.10e+19                                                                                                        

=== Evaluating checkpoint_step1740.pt | step=1740 | params=24,985,344 ===
[✓] step=1740 | avg_loss=44.0646 | batches=250 | tokens=304327 | ppl=1.37e+19                                                                                                        
[skip] checkpoint_step176.pt: checkpoint vocab 60004 != tokenizer vocab 24000

=== Evaluating checkpoint_step1802.pt | step=1802 | params=24,985,344 ===
[✓] step=1802 | avg_loss=43.3464 | batches=250 | tokens=304327 | ppl=6.68e+18                                                                                                        

=== Evaluating checkpoint_step1865.pt | step=1865 | params=24,985,344 ===
[✓] step=1865 | avg_loss=42.5606 | batches=250 | tokens=304327 | ppl=3.05e+18                                                                                                        
[skip] checkpoint_step192.pt: checkpoint vocab 60004 != tokenizer vocab 24000

=== Evaluating checkpoint_step1928.pt | step=1928 | params=24,985,344 ===
[✓] step=1928 | avg_loss=41.7266 | batches=250 | tokens=304327 | ppl=1.32e+18                                                                                                        

=== Evaluating checkpoint_step1991.pt | step=1991 | params=24,985,344 ===
[✓] step=1991 | avg_loss=41.7266 | batches=250 | tokens=304327 | ppl=1.32e+18                                                                                                        

=== Evaluating checkpoint_step2054.pt | step=2054 | params=24,985,344 ===
[✓] step=2054 | avg_loss=40.9691 | batches=250 | tokens=304327 | ppl=6.20e+17                                                                                                        
[!] Failed to load checkpoint_step208.pt: [Errno 2] No such file or directory: 'artifacts/zia_ift_fixed4k/checkpoints\\checkpoint_step208.pt'. Skipping.

=== Evaluating checkpoint_step2117.pt | step=2117 | params=24,985,344 ===
[✓] step=2117 | avg_loss=40.2757 | batches=250 | tokens=304327 | ppl=3.10e+17                                                                                                        

=== Evaluating checkpoint_step2180.pt | step=2180 | params=24,985,344 ===
[✓] step=2180 | avg_loss=39.5593 | batches=250 | tokens=304327 | ppl=1.51e+17                                                                                                        
[!] Failed to load checkpoint_step224.pt: [Errno 2] No such file or directory: 'artifacts/zia_ift_fixed4k/checkpoints\\checkpoint_step224.pt'. Skipping.

=== Evaluating checkpoint_step2242.pt | step=2242 | params=24,985,344 ===
[✓] step=2242 | avg_loss=38.9210 | batches=250 | tokens=304327 | ppl=8.00e+16                                                                                                        
[!] Failed to load checkpoint_step240.pt: [Errno 2] No such file or directory: 'artifacts/zia_ift_fixed4k/checkpoints\\checkpoint_step240.pt'. Skipping.

=== Evaluating checkpoint_step296.pt | step=296 | params=24,985,344 ===
[✓] step=296 | avg_loss=187.6432 | batches=250 | tokens=304327 | ppl=inf                                                                                                             
[!] Failed to load checkpoint_step32.pt: [Errno 2] No such file or directory: 'artifacts/zia_ift_fixed4k/checkpoints\\checkpoint_step32.pt'. Skipping.

=== Evaluating checkpoint_step356.pt | step=356 | params=24,985,344 ===
Traceback (most recent call last):                                                                                                                                                   
  File "D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py", line 195, in <module>
    n_tokens = int(token_mask.sum().item())
                   ^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> ls .\artifacts\zia_ift_fixed4k\checkpoints\


    Directory: D:\aria\aria_ai\aria_ai_assistant\artifacts\zia_ift_fixed4k\checkpoints


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        23-11-2025     14:04                Old
-a----        23-11-2025     03:48       99978114 checkpoint_step1049.pt
-a----        23-11-2025     04:08       99978114 checkpoint_step1112.pt
-a----        23-11-2025     04:28       99978114 checkpoint_step1175.pt
-a----        23-11-2025     04:49       99978114 checkpoint_step1238.pt
-a----        23-11-2025     05:09       99978114 checkpoint_step1299.pt
-a----        23-11-2025     05:29       99978114 checkpoint_step1362.pt
-a----        23-11-2025     05:49       99978114 checkpoint_step1425.pt
-a----        23-11-2025     06:09       99978114 checkpoint_step1488.pt
-a----        23-11-2025     06:30       99978114 checkpoint_step1551.pt
-a----        23-11-2025     06:50       99978114 checkpoint_step1614.pt
-a----        23-11-2025     07:10       99978114 checkpoint_step1677.pt
-a----        23-11-2025     07:30       99978114 checkpoint_step1740.pt
-a----        23-11-2025     07:51       99978114 checkpoint_step1802.pt
-a----        23-11-2025     08:11       99978114 checkpoint_step1865.pt
-a----        23-11-2025     08:31       99978114 checkpoint_step1928.pt
-a----        23-11-2025     08:51       99978114 checkpoint_step1991.pt
-a----        23-11-2025     09:12       99978114 checkpoint_step2054.pt
-a----        23-11-2025     09:32       99978114 checkpoint_step2117.pt
-a----        23-11-2025     09:52       99978114 checkpoint_step2180.pt
-a----        23-11-2025     10:12       99978114 checkpoint_step2242.pt






------------------------------------------------------------------------------------------------------------------------------------
___________________________________________________________________________________________________________________________________

     PHASE - 2 - IFT TRAINING From 2242 Checkpoint - step=2242 | avg_loss=38.9210 | batches=250 | tokens=304327 | ppl=8.00e+16  

------------------------------------------------------------------------------------------------------------------------------------
___________________________________________________________________________________________________________________________________



(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\my_datasets\code\train_realign_v10.py
[i] Device: cuda
[i] Vocab: 24000 | Pad ID: 0
[i] Loading weights from artifacts/zia_ift_fixed4k/checkpoints/checkpoint_step2242.pt...
[i] Loaded 101 layers. Skipped 0 mismatches.
[i] Found 1397 shards. Using 1397.
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2296.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2353.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2413.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2473.pt                                                                                                  
Shard 9/1397: 100%|███████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:27:01<00:00, 20.89s/it, loss=38.6565, lr=2.50e-04, step=2479] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2532.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2592.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2652.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2712.pt                                                                                                  
Shard 10/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:24:37<00:00, 20.31s/it, loss=37.9878, lr=2.50e-04, step=2719] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2771.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2831.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2891.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step2951.pt                                                                                                  
Shard 11/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:24:51<00:00, 20.37s/it, loss=36.8933, lr=2.50e-04, step=2959] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3011.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3071.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3131.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3191.pt                                                                                                  
Shard 12/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:24:30<00:00, 20.28s/it, loss=34.9814, lr=2.50e-04, step=3199] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3251.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3311.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3371.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3431.pt                                                                                                  
Shard 13/1397:  83%|███████████████████████████████████████████████████████████████████████▏              | 207/250 [1:10:32<14:39, 20.45s/it, loss=33.1617, lr=2.50e-04, step=3439] 
Traceback (most recent call last):
  File "D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\train_realign_v10.py", line 286, in <module>
    train()
  File "D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\train_realign_v10.py", line 250, in train
    scaler.scale(loss).backward()
  File "D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\torch\_tensor.py", line 581, in backward
    torch.autograd.backward(
  File "D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\torch\autograd\__init__.py", line 347, in backward
    _engine_run_backward(
  File "D:\aria\aria_ai\aria_ai_assistant\.venv\Lib\site-packages\torch\autograd\graph.py", line 825, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 
 *  History restored 

PS D:\aria\aria_ai\aria_ai_assistant> & D:\aria\aria_ai\aria_ai_assistant\.venv\Scripts\Activate.ps1
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\my_datasets\code\eval_chkpt3.py      
[i] Device: cuda
[i] Loaded SentencePiece tokenizer | vocab=24000 | pad_id=0
[i] Loading val: datasets/processed/test_openorca/val/shard_000
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:42: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.ids = torch.load(ids_path)
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:43: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.lbl = torch.load(lbl_path)
[i] Found 20 checkpoints
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:129: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ck = torch.load(ckpt, map_location="cpu")

=== Evaluating checkpoint_step2296.pt | step=2296 | params=24,985,344 ===
[✓] step=2296 | avg_loss=38.9210 | batches=250 | tokens=304327 | ppl=8.00e+16                                                                                                        

=== Evaluating checkpoint_step2353.pt | step=2353 | params=24,985,344 ===
[✓] step=2353 | avg_loss=40.7996 | batches=250 | tokens=304327 | ppl=5.24e+17                                                                                                        

=== Evaluating checkpoint_step2413.pt | step=2413 | params=24,985,344 ===
[✓] step=2413 | avg_loss=40.7231 | batches=250 | tokens=304327 | ppl=4.85e+17                                                                                                        

=== Evaluating checkpoint_step2473.pt | step=2473 | params=24,985,344 ===
[✓] step=2473 | avg_loss=40.7231 | batches=250 | tokens=304327 | ppl=4.85e+17                                                                                                        

=== Evaluating checkpoint_step2532.pt | step=2532 | params=24,985,344 ===
[✓] step=2532 | avg_loss=40.8121 | batches=250 | tokens=304327 | ppl=5.30e+17                                                                                                        

=== Evaluating checkpoint_step2592.pt | step=2592 | params=24,985,344 ===
[✓] step=2592 | avg_loss=39.9374 | batches=250 | tokens=304327 | ppl=2.21e+17                                                                                                        

=== Evaluating checkpoint_step2652.pt | step=2652 | params=24,985,344 ===
[✓] step=2652 | avg_loss=39.4379 | batches=250 | tokens=304327 | ppl=1.34e+17                                                                                                        

=== Evaluating checkpoint_step2712.pt | step=2712 | params=24,985,344 ===
[✓] step=2712 | avg_loss=39.4379 | batches=250 | tokens=304327 | ppl=1.34e+17                                                                                                        

=== Evaluating checkpoint_step2771.pt | step=2771 | params=24,985,344 ===
[✓] step=2771 | avg_loss=38.9861 | batches=250 | tokens=304327 | ppl=8.54e+16                                                                                                        

=== Evaluating checkpoint_step2831.pt | step=2831 | params=24,985,344 ===
[✓] step=2831 | avg_loss=38.4919 | batches=250 | tokens=304327 | ppl=5.21e+16                                                                                                        

=== Evaluating checkpoint_step2891.pt | step=2891 | params=24,985,344 ===
[✓] step=2891 | avg_loss=37.6816 | batches=250 | tokens=304327 | ppl=2.32e+16                                                                                                        

=== Evaluating checkpoint_step2951.pt | step=2951 | params=24,985,344 ===
[✓] step=2951 | avg_loss=37.6816 | batches=250 | tokens=304327 | ppl=2.32e+16                                                                                                        

=== Evaluating checkpoint_step3011.pt | step=3011 | params=24,985,344 ===
[✓] step=3011 | avg_loss=36.9300 | batches=250 | tokens=304327 | ppl=1.09e+16                                                                                                        

=== Evaluating checkpoint_step3071.pt | step=3071 | params=24,985,344 ===
[✓] step=3071 | avg_loss=36.3927 | batches=250 | tokens=304327 | ppl=6.38e+15                                                                                                        

=== Evaluating checkpoint_step3131.pt | step=3131 | params=24,985,344 ===
[✓] step=3131 | avg_loss=36.1824 | batches=250 | tokens=304327 | ppl=5.17e+15                                                                                                        
=== Evaluating checkpoint_step3191.pt | step=3191 | params=24,985,344 ===
[✓] step=3191 | avg_loss=36.1824 | batches=250 | tokens=304327 | ppl=5.17e+15                                                                                                        

=== Evaluating checkpoint_step3251.pt | step=3251 | params=24,985,344 ===
[✓] step=3251 | avg_loss=35.8788 | batches=250 | tokens=304327 | ppl=3.82e+15                                                                                                        

=== Evaluating checkpoint_step3311.pt | step=3311 | params=24,985,344 ===
[✓] step=3311 | avg_loss=35.5464 | batches=250 | tokens=304327 | ppl=2.74e+15                                                                                                        

=== Evaluating checkpoint_step3371.pt | step=3371 | params=24,985,344 ===
[✓] step=3371 | avg_loss=35.1056 | batches=250 | tokens=304327 | ppl=1.76e+15                                                                                                        

=== Evaluating checkpoint_step3431.pt | step=3431 | params=24,985,344 ===
[✓] step=3431 | avg_loss=35.1056 | batches=250 | tokens=304327 | ppl=1.76e+15                                                                                                        

=== BEST CHECKPOINT ===
checkpoint_step3371.pt  step=3371  avg_loss=35.1056  tokens=304327
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 



------------------------------------------------------------------------------------------------------------------------------------
___________________________________________________________________________________________________________________________________

     PHASE - 3 - IFT TRAINING From 3431 Checkpoint - step=3431 | avg_loss=35.1056 | batches=250 | tokens=304327 | ppl=1.76e 

------------------------------------------------------------------------------------------------------------------------------------
___________________________________________________________________________________________________________________________________







(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\my_datasets\code\train_realign_v10.py
[i] Device: cuda
[i] Vocab: 24000 | Pad ID: 0
[i] Loading weights from artifacts/zia_ift_fixed4k/checkpoints/checkpoint_step3431.pt...
[i] Loaded 101 layers. Skipped 0 mismatches.
[i] Found 1397 shards. Using 1397.
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3483.pt
Shard 13/1397: 100%|████████████████████████████████████████████████████████████████████████████████████████| 250/250 [23:35<00:00,  5.66s/it, loss=31.4341, lr=2.50e-04, step=3439] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3535.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3587.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3634.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3679.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3726.pt                                                                                                  
Shard 14/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:45:03<00:00, 25.22s/it, loss=36.2935, lr=2.50e-04, step=3679] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3772.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3818.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3865.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3911.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step3956.pt                                                                                                  
Shard 15/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:49:56<00:00, 26.39s/it, loss=33.1931, lr=2.50e-04, step=3919] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4002.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4048.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4094.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4140.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4186.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4232.pt                                                                                                  
Shard 16/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:49:03<00:00, 26.17s/it, loss=33.2021, lr=2.50e-04, step=4239] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4278.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4324.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4370.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4416.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4462.pt                                                                                                  
Shard 17/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:49:05<00:00, 26.18s/it, loss=31.2241, lr=2.50e-04, step=4479] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4508.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4554.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4600.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4646.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4692.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4738.pt                                                                                                  
Shard 18/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:49:09<00:00, 26.20s/it, loss=30.9247, lr=2.50e-04, step=4719] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4784.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4830.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4877.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4923.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step4970.pt                                                                                                  
Shard 19/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:48:59<00:00, 26.16s/it, loss=30.7238, lr=2.50e-04, step=4959] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5016.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5062.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5109.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5156.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5203.pt                                                                                                  
Shard 20/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:47:57<00:00, 25.91s/it, loss=28.3613, lr=2.50e-04, step=5199] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5250.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5297.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5344.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5391.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5438.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5485.pt                                                                                                  
Shard 21/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:47:36<00:00, 25.83s/it, loss=26.2991, lr=2.50e-04, step=5439] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5532.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5579.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5626.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5673.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5720.pt                                                                                                  
Shard 22/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:47:36<00:00, 25.82s/it, loss=29.5383, lr=2.50e-04, step=5679] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5767.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5814.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5861.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5908.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step5955.pt                                                                                                  
Shard 23/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:47:48<00:00, 25.87s/it, loss=27.4791, lr=2.50e-04, step=5919] 
Shard 24/1397:   0%|                                                                                                  





(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\my_datasets\code\eval_chkpt3.py      
[i] Device: cuda
[i] Loaded SentencePiece tokenizer | vocab=24000 | pad_id=0
[i] Loading val: datasets/processed/test_openorca/val/shard_000
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:42: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.ids = torch.load(ids_path)
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:43: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.lbl = torch.load(lbl_path)
[i] Found 54 checkpoints
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:129: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ck = torch.load(ckpt, map_location="cpu")

=== Evaluating checkpoint_step3483.pt | step=3483 | params=24,985,344 ===
[✓] step=3483 | avg_loss=38.0833 | batches=250 | tokens=304327 | ppl=3.46e+16                                                                                                        

=== Evaluating checkpoint_step3535.pt | step=3535 | params=24,985,344 ===
[✓] step=3535 | avg_loss=37.7623 | batches=250 | tokens=304327 | ppl=2.51e+16                                                                                                        

=== Evaluating checkpoint_step3587.pt | step=3587 | params=24,985,344 ===
[✓] step=3587 | avg_loss=37.7623 | batches=250 | tokens=304327 | ppl=2.51e+16                                                                                                        

=== Evaluating checkpoint_step3634.pt | step=3634 | params=24,985,344 ===
[✓] step=3634 | avg_loss=37.1349 | batches=250 | tokens=304327 | ppl=1.34e+16                                                                                                        

=== Evaluating checkpoint_step3679.pt | step=3679 | params=24,985,344 ===
[✓] step=3679 | avg_loss=37.1349 | batches=250 | tokens=304327 | ppl=1.34e+16                                                                                                        

=== Evaluating checkpoint_step3726.pt | step=3726 | params=24,985,344 ===
[✓] step=3726 | avg_loss=36.7932 | batches=250 | tokens=304327 | ppl=9.53e+15                                                                                                        

=== Evaluating checkpoint_step3772.pt | step=3772 | params=24,985,344 ===
[✓] step=3772 | avg_loss=36.3259 | batches=250 | tokens=304327 | ppl=5.97e+15                                                                                                        

=== Evaluating checkpoint_step3818.pt | step=3818 | params=24,985,344 ===
[✓] step=3818 | avg_loss=36.3259 | batches=250 | tokens=304327 | ppl=5.97e+15                                                                                                        

=== Evaluating checkpoint_step3865.pt | step=3865 | params=24,985,344 ===
[✓] step=3865 | avg_loss=35.7541 | batches=250 | tokens=304327 | ppl=3.37e+15                                                                                                        

=== Evaluating checkpoint_step3911.pt | step=3911 | params=24,985,344 ===
[✓] step=3911 | avg_loss=35.7541 | batches=250 | tokens=304327 | ppl=3.37e+15                                                                                                        

=== Evaluating checkpoint_step3956.pt | step=3956 | params=24,985,344 ===
[✓] step=3956 | avg_loss=35.1440 | batches=250 | tokens=304327 | ppl=1.83e+15                                                                                                        

=== Evaluating checkpoint_step4002.pt | step=4002 | params=24,985,344 ===
[✓] step=4002 | avg_loss=34.5834 | batches=250 | tokens=304327 | ppl=1.05e+15                                                                                                        

=== Evaluating checkpoint_step4048.pt | step=4048 | params=24,985,344 ===
[✓] step=4048 | avg_loss=34.5834 | batches=250 | tokens=304327 | ppl=1.05e+15                                                                                                        

=== Evaluating checkpoint_step4094.pt | step=4094 | params=24,985,344 ===
[✓] step=4094 | avg_loss=33.9407 | batches=250 | tokens=304327 | ppl=5.50e+14                                                                                                        

=== Evaluating checkpoint_step4140.pt | step=4140 | params=24,985,344 ===
[✓] step=4140 | avg_loss=33.9407 | batches=250 | tokens=304327 | ppl=5.50e+14                                                                                                        

=== Evaluating checkpoint_step4186.pt | step=4186 | params=24,985,344 ===
[✓] step=4186 | avg_loss=33.4428 | batches=250 | tokens=304327 | ppl=3.34e+14          

=== Evaluating checkpoint_step4232.pt | step=4232 | params=24,985,344 ===
[✓] step=4232 | avg_loss=33.4428 | batches=250 | tokens=304327 | ppl=3.34e+14                                                                                                        

=== Evaluating checkpoint_step4278.pt | step=4278 | params=24,985,344 ===
[✓] step=4278 | avg_loss=33.0700 | batches=250 | tokens=304327 | ppl=2.30e+14                                                                                                        

=== Evaluating checkpoint_step4324.pt | step=4324 | params=24,985,344 ===
[✓] step=4324 | avg_loss=32.8447 | batches=250 | tokens=304327 | ppl=1.84e+14                                                                                                        

=== Evaluating checkpoint_step4370.pt | step=4370 | params=24,985,344 ===
[✓] step=4370 | avg_loss=32.8447 | batches=250 | tokens=304327 | ppl=1.84e+14                                                                                                        

=== Evaluating checkpoint_step4416.pt | step=4416 | params=24,985,344 ===
[✓] step=4416 | avg_loss=32.7680 | batches=250 | tokens=304327 | ppl=1.70e+14                                                                                                        

=== Evaluating checkpoint_step4462.pt | step=4462 | params=24,985,344 ===
[✓] step=4462 | avg_loss=32.7680 | batches=250 | tokens=304327 | ppl=1.70e+14                                                                                                        

=== Evaluating checkpoint_step4508.pt | step=4508 | params=24,985,344 ===
[✓] step=4508 | avg_loss=32.4809 | batches=250 | tokens=304327 | ppl=1.28e+14                                                                                                        

=== Evaluating checkpoint_step4554.pt | step=4554 | params=24,985,344 ===
[✓] step=4554 | avg_loss=32.4809 | batches=250 | tokens=304327 | ppl=1.28e+14                                                                                                        

=== Evaluating checkpoint_step4600.pt | step=4600 | params=24,985,344 ===
[✓] step=4600 | avg_loss=32.2188 | batches=250 | tokens=304327 | ppl=9.83e+13                                                                                                        

=== Evaluating checkpoint_step4646.pt | step=4646 | params=24,985,344 ===
[✓] step=4646 | avg_loss=31.9880 | batches=250 | tokens=304327 | ppl=7.80e+13                                                                                                        

=== Evaluating checkpoint_step4692.pt | step=4692 | params=24,985,344 ===
[✓] step=4692 | avg_loss=31.9880 | batches=250 | tokens=304327 | ppl=7.80e+13                                                                                                        

=== Evaluating checkpoint_step4738.pt | step=4738 | params=24,985,344 ===
[✓] step=4738 | avg_loss=31.7392 | batches=250 | tokens=304327 | ppl=6.08e+13                                                                                                        

=== Evaluating checkpoint_step4784.pt | step=4784 | params=24,985,344 ===
[✓] step=4784 | avg_loss=31.7392 | batches=250 | tokens=304327 | ppl=6.08e+13                                                                                                        

=== Evaluating checkpoint_step4830.pt | step=4830 | params=24,985,344 ===
[✓] step=4830 | avg_loss=31.4657 | batches=250 | tokens=304327 | ppl=4.63e+13                                                                                                        

=== Evaluating checkpoint_step4877.pt | step=4877 | params=24,985,344 ===
[✓] step=4877 | avg_loss=31.4657 | batches=250 | tokens=304327 | ppl=4.63e+13                                                                                                        

=== Evaluating checkpoint_step4923.pt | step=4923 | params=24,985,344 ===
[✓] step=4923 | avg_loss=31.2302 | batches=250 | tokens=304327 | ppl=3.66e+13                                                                                                        

=== Evaluating checkpoint_step4970.pt | step=4970 | params=24,985,344 ===
[✓] step=4970 | avg_loss=30.9842 | batches=250 | tokens=304327 | ppl=2.86e+13                                                                                                        

=== Evaluating checkpoint_step5016.pt | step=5016 | params=24,985,344 ===
[✓] step=5016 | avg_loss=30.9842 | batches=250 | tokens=304327 | ppl=2.86e+13                                                                                                        

=== Evaluating checkpoint_step5062.pt | step=5062 | params=24,985,344 ===
[✓] step=5062 | avg_loss=30.8009 | batches=250 | tokens=304327 | ppl=2.38e+13                                                                                                        

=== Evaluating checkpoint_step5109.pt | step=5109 | params=24,985,344 ===
[✓] step=5109 | avg_loss=30.8009 | batches=250 | tokens=304327 | ppl=2.38e+13                                                                                                        

=== Evaluating checkpoint_step5156.pt | step=5156 | params=24,985,344 ===
[✓] step=5156 | avg_loss=30.6566 | batches=250 | tokens=304327 | ppl=2.06e+13                                                                                                        

=== Evaluating checkpoint_step5203.pt | step=5203 | params=24,985,344 ===
[✓] step=5203 | avg_loss=30.4584 | batches=250 | tokens=304327 | ppl=1.69e+13                                                                                                        

=== Evaluating checkpoint_step5250.pt | step=5250 | params=24,985,344 ===
[✓] step=5250 | avg_loss=30.4584 | batches=250 | tokens=304327 | ppl=1.69e+13                                                                                                        

=== Evaluating checkpoint_step5297.pt | step=5297 | params=24,985,344 ===
[✓] step=5297 | avg_loss=30.2443 | batches=250 | tokens=304327 | ppl=1.36e+13                                                                                                        

=== Evaluating checkpoint_step5344.pt | step=5344 | params=24,985,344 ===
[✓] step=5344 | avg_loss=30.2443 | batches=250 | tokens=304327 | ppl=1.36e+13                                                                                                        

=== Evaluating checkpoint_step5391.pt | step=5391 | params=24,985,344 ===
[✓] step=5391 | avg_loss=30.0160 | batches=250 | tokens=304327 | ppl=1.09e+13                                                                                                        

=== Evaluating checkpoint_step5438.pt | step=5438 | params=24,985,344 ===
[✓] step=5438 | avg_loss=30.0160 | batches=250 | tokens=304327 | ppl=1.09e+13                                                                                                        

=== Evaluating checkpoint_step5485.pt | step=5485 | params=24,985,344 ===
[✓] step=5485 | avg_loss=29.8798 | batches=250 | tokens=304327 | ppl=9.48e+12                                                                                                        

=== Evaluating checkpoint_step5532.pt | step=5532 | params=24,985,344 ===
[✓] step=5532 | avg_loss=29.6510 | batches=250 | tokens=304327 | ppl=7.54e+12                                                                                                        

=== Evaluating checkpoint_step5579.pt | step=5579 | params=24,985,344 ===
[✓] step=5579 | avg_loss=29.6510 | batches=250 | tokens=304327 | ppl=7.54e+12                                                                                                        

=== Evaluating checkpoint_step5626.pt | step=5626 | params=24,985,344 ===
[✓] step=5626 | avg_loss=29.4310 | batches=250 | tokens=304327 | ppl=6.05e+12                                                                                                        

=== Evaluating checkpoint_step5673.pt | step=5673 | params=24,985,344 ===
[✓] step=5673 | avg_loss=29.4310 | batches=250 | tokens=304327 | ppl=6.05e+12                                                                                                        

=== Evaluating checkpoint_step5720.pt | step=5720 | params=24,985,344 ===
[✓] step=5720 | avg_loss=29.2334 | batches=250 | tokens=304327 | ppl=4.96e+12                                                                                                        

=== Evaluating checkpoint_step5767.pt | step=5767 | params=24,985,344 ===
[✓] step=5767 | avg_loss=29.0719 | batches=250 | tokens=304327 | ppl=4.22e+12                                                                                                        

=== Evaluating checkpoint_step5814.pt | step=5814 | params=24,985,344 ===
[✓] step=5814 | avg_loss=29.0719 | batches=250 | tokens=304327 | ppl=4.22e+12                                                                                                        

=== Evaluating checkpoint_step5861.pt | step=5861 | params=24,985,344 ===
[✓] step=5861 | avg_loss=28.8819 | batches=250 | tokens=304327 | ppl=3.49e+12                                                                                                        

=== Evaluating checkpoint_step5908.pt | step=5908 | params=24,985,344 ===
[✓] step=5908 | avg_loss=28.8819 | batches=250 | tokens=304327 | ppl=3.49e+12                                                                                                        

=== Evaluating checkpoint_step5955.pt | step=5955 | params=24,985,344 ===
[✓] step=5955 | avg_loss=28.6766 | batches=250 | tokens=304327 | ppl=2.84e+12                                                                                                        

=== BEST CHECKPOINT ===
checkpoint_step5955.pt  step=5955  avg_loss=28.6766  tokens=304327
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 



------------------------------------------------------------------------------------------------------------------------------------
___________________________________________________________________________________________________________________________________

     PHASE - 4 - IFT TRAINING From 5955 Checkpoint - step=5955 | avg_loss=28.6766 | batches=250 | tokens=304327 | ppl=2.84e+12 

------------------------------------------------------------------------------------------------------------------------------------
___________________________________________________________________________________________________________________________________




(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\my_datasets\code\train_realign_v10.py
[i] Device: cuda
[i] Vocab: 24000 | Pad ID: 0
[i] Loading weights from artifacts/zia_ift_fixed4k/checkpoints/checkpoint_step5955.pt...
[i] Loaded 101 layers. Skipped 0 mismatches.
[i] Found 1397 shards. Using 1397.
Shard 23/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [10:01<00:00,  2.41s/it]
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step6063.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step6168.pt                                                                                                  
Shard 24/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:11:40<00:00, 17.20s/it, loss=31.5145, lr=2.50e-04, step=6239] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step6274.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step6379.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step6485.pt                                                                                                  
Shard 25/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:11:30<00:00, 17.16s/it, loss=30.4538, lr=2.50e-04, step=6479] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step6590.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step6696.pt                                                                                                  
Shard 26/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:11:35<00:00, 17.18s/it, loss=26.6495, lr=2.50e-04, step=6719] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step6801.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step6907.pt                                                                                                  
Shard 27/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:11:47<00:00, 17.23s/it, loss=26.6440, lr=2.50e-04, step=6959] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step7012.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step7118.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step7224.pt                                                                                                  
Shard 28/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:11:22<00:00, 17.13s/it, loss=26.1753, lr=2.50e-04, step=7199] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step7330.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step7434.pt                                                                                                  
Shard 29/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:13:24<00:00, 17.62s/it, loss=22.9743, lr=2.50e-04, step=7439] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step7532.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step7631.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step7729.pt                                                                                                  
Shard 30/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:17:06<00:00, 18.51s/it, loss=27.4276, lr=2.50e-04, step=7679] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step7828.pt
Shard 31/1397:  73%|████████████████████████████████████████████████████████████████▍                       | 183/250 [56:38<20:44, 18.57s/it, loss=24.1580, lr=2.50e-04, step=7919] 


(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\my_datasets\code\eval_chkpt3.py
[i] Device: cuda
[i] Loaded SentencePiece tokenizer | vocab=24000 | pad_id=0
[i] Loading val: datasets/processed/test_openorca/val/shard_000
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:42: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.ids = torch.load(ids_path)
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:43: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.lbl = torch.load(lbl_path)
[i] Found 18 checkpoints
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:129: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ck = torch.load(ckpt, map_location="cpu")

=== Evaluating checkpoint_step6063.pt | step=6063 | params=24,985,344 ===
[✓] step=6063 | avg_loss=33.1587 | batches=250 | tokens=304327 | ppl=2.52e+14                                                                                                        

=== Evaluating checkpoint_step6168.pt | step=6168 | params=24,985,344 ===
[✓] step=6168 | avg_loss=32.5190 | batches=250 | tokens=304327 | ppl=1.33e+14                                                                                                        

=== Evaluating checkpoint_step6274.pt | step=6274 | params=24,985,344 ===
[✓] step=6274 | avg_loss=31.2163 | batches=250 | tokens=304327 | ppl=3.61e+13                                                                                                        

=== Evaluating checkpoint_step6379.pt | step=6379 | params=24,985,344 ===
[✓] step=6379 | avg_loss=30.7447 | batches=250 | tokens=304327 | ppl=2.25e+13                                                                                                        

=== Evaluating checkpoint_step6485.pt | step=6485 | params=24,985,344 ===
[✓] step=6485 | avg_loss=29.7957 | batches=250 | tokens=304327 | ppl=8.71e+12                                                                                                        

=== Evaluating checkpoint_step6590.pt | step=6590 | params=24,985,344 ===
[✓] step=6590 | avg_loss=29.2051 | batches=250 | tokens=304327 | ppl=4.83e+12                                                                                                        

=== Evaluating checkpoint_step6696.pt | step=6696 | params=24,985,344 ===
[✓] step=6696 | avg_loss=28.6336 | batches=250 | tokens=304327 | ppl=2.73e+12                                                                                                        

=== Evaluating checkpoint_step6801.pt | step=6801 | params=24,985,344 ===
[✓] step=6801 | avg_loss=28.2724 | batches=250 | tokens=304327 | ppl=1.90e+12                                                                                                        

=== Evaluating checkpoint_step6907.pt | step=6907 | params=24,985,344 ===
[✓] step=6907 | avg_loss=28.8065 | batches=250 | tokens=304327 | ppl=3.24e+12                                                                                                        

=== Evaluating checkpoint_step7012.pt | step=7012 | params=24,985,344 ===
[✓] step=7012 | avg_loss=28.5158 | batches=250 | tokens=304327 | ppl=2.42e+12                                                                                                        

=== Evaluating checkpoint_step7118.pt | step=7118 | params=24,985,344 ===
[✓] step=7118 | avg_loss=28.0692 | batches=250 | tokens=304327 | ppl=1.55e+12                                                                                                        

=== Evaluating checkpoint_step7224.pt | step=7224 | params=24,985,344 ===
[✓] step=7224 | avg_loss=27.6077 | batches=250 | tokens=304327 | ppl=9.77e+11                                                                                                        

=== Evaluating checkpoint_step7330.pt | step=7330 | params=24,985,344 ===
[✓] step=7330 | avg_loss=27.3631 | batches=250 | tokens=304327 | ppl=7.65e+11                                                                                                        

=== Evaluating checkpoint_step7434.pt | step=7434 | params=24,985,344 ===
[✓] step=7434 | avg_loss=27.1910 | batches=250 | tokens=304327 | ppl=6.44e+11                                                                                                        

=== Evaluating checkpoint_step7532.pt | step=7532 | params=24,985,344 ===
[✓] step=7532 | avg_loss=26.7582 | batches=250 | tokens=304327 | ppl=4.18e+11                                                                                                        

=== Evaluating checkpoint_step7631.pt | step=7631 | params=24,985,344 ===
[✓] step=7631 | avg_loss=26.6640 | batches=250 | tokens=304327 | ppl=3.80e+11                                                                                                        

=== Evaluating checkpoint_step7729.pt | step=7729 | params=24,985,344 ===
[✓] step=7729 | avg_loss=26.4807 | batches=250 | tokens=304327 | ppl=3.17e+11                                                                                                        

=== Evaluating checkpoint_step7828.pt | step=7828 | params=24,985,344 ===
[✓] step=7828 | avg_loss=26.3336 | batches=250 | tokens=304327 | ppl=2.73e+11                                                                                                        

=== BEST CHECKPOINT ===
checkpoint_step7828.pt  step=7828  avg_loss=26.3336  tokens=304327
(.venv) PS D:\aria\aria_ai\aria_ai_assistant> 





------------------------------------------------------------------------------------------------------------------------------------
___________________________________________________________________________________________________________________________________

     PHASE - 5 - IFT TRAINING From 7828 Checkpoint - step=7828 | avg_loss=26.3336 | batches=250 | tokens=304327 | ppl=2.73e+11 

------------------------------------------------------------------------------------------------------------------------------------
___________________________________________________________________________________________________________________________________









.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\my_datasets\code\train_realign_v10.py
[i] Device: cuda
[i] Vocab: 24000 | Pad ID: 0
[i] Loading weights from artifacts/zia_ift_fixed4k/checkpoints/checkpoint_step7828.pt...
[i] Loaded 101 layers. Skipped 0 mismatches.
[i] Found 1397 shards. Using 1397.
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step7940.pt
Shard 31/1397: 100%|████████████████████████████████████████████████████████████████████████████████████████| 250/250 [44:56<00:00, 10.79s/it, loss=26.8599, lr=2.50e-04, step=7919]
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step8051.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step8163.pt
████████████████████████████████████████████████████████████████████████████████| 250/250 [22:28<00:00,  5.40s/it, loss=26.6536, lr=2.50e-04, step=8239]
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step8271.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step8376.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step8480.pt
Shard 33/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:12:30<00:00, 17.40s/it, loss=24.6564, lr=2.50e-04, step=8479]
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step8584.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step8688.pt
Shard 34/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:12:54<00:00, 17.50s/it, loss=26.4832, lr=2.50e-04, step=8719]
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step8792.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step8897.pt                                                                                                  
Shard 35/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:12:39<00:00, 17.44s/it, loss=23.6022, lr=2.50e-04, step=8959] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step9002.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step9107.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step9211.pt                                                                                                  
Shard 36/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:12:45<00:00, 17.46s/it, loss=24.7268, lr=2.50e-04, step=9199] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step9315.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step9419.pt                                                                                                  
Shard 37/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:12:46<00:00, 17.47s/it, loss=23.4078, lr=2.50e-04, step=9439] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step9524.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step9628.pt                                                                                                  
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step9732.pt                                                                                                  
Shard 38/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:12:50<00:00, 17.48s/it, loss=21.6844, lr=2.50e-04, step=9679] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step9837.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step9941.pt                                                                                                  
Shard 39/1397: 100%|██████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:12:48<00:00, 17.47s/it, loss=22.5142, lr=2.50e-04, step=9919] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step10046.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step10150.pt                                                                                                 
Shard 40/1397: 100%|█████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:12:45<00:00, 17.46s/it, loss=21.0040, lr=2.50e-04, step=10239] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step10255.pt
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step10360.pt                                                                                                 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step10464.pt                                                                                                 
Shard 41/1397: 100%|█████████████████████████████████████████████████████████████████████████████████████| 250/250 [1:12:46<00:00, 17.46s/it, loss=20.5512, lr=2.50e-04, step=10479] 
[💾] Saved checkpoint: artifacts/zia_ift_fixed4k/checkpoints\checkpoint_step10568.pt
Shard 42/1397:  31%|███████████████████████████                                                             | 77/250 [22:29<50:07, 17.39s/it, loss=23.2880, lr=2.50e-04, step=10559] 



(.venv) PS D:\aria\aria_ai\aria_ai_assistant> python .\my_datasets\code\eval_chkpt3.py
[i] Device: cuda
[i] Loaded SentencePiece tokenizer | vocab=24000 | pad_id=0
[i] Loading val: datasets/processed/test_openorca/val/shard_000
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:42: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.ids = torch.load(ids_path)
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:43: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.lbl = torch.load(lbl_path)
[i] Found 27 checkpoints
D:\aria\aria_ai\aria_ai_assistant\my_datasets\code\eval_chkpt3.py:129: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ck = torch.load(ckpt, map_location="cpu")

=== Evaluating checkpoint_step10046.pt | step=10046 | params=24,985,344 ===
[✓] step=10046 | avg_loss=23.3261 | batches=100 | tokens=30467 | ppl=1.35e+10                                                                                                        

=== Evaluating checkpoint_step10150.pt | step=10150 | params=24,985,344 ===
[✓] step=10150 | avg_loss=23.1146 | batches=100 | tokens=30467 | ppl=1.09e+10                                                                                                        

=== Evaluating checkpoint_step10255.pt | step=10255 | params=24,985,344 ===
[✓] step=10255 | avg_loss=22.8882 | batches=100 | tokens=30467 | ppl=8.71e+09                                                                                                        

=== Evaluating checkpoint_step10360.pt | step=10360 | params=24,985,344 ===
[✓] step=10360 | avg_loss=22.8038 | batches=100 | tokens=30467 | ppl=8.01e+09                                                                                                        

=== Evaluating checkpoint_step10464.pt | step=10464 | params=24,985,344 ===
[✓] step=10464 | avg_loss=22.7618 | batches=100 | tokens=30467 | ppl=7.68e+09                                                                                                        

=== Evaluating checkpoint_step10568.pt | step=10568 | params=24,985,344 ===
[✓] step=10568 | avg_loss=22.4348 | batches=100 | tokens=30467 | ppl=5.54e+09                                                                                                        

=== Evaluating checkpoint_step7828.pt | step=7828 | params=24,985,344 ===
[✓] step=7828 | avg_loss=26.3593 | batches=100 | tokens=30467 | ppl=2.80e+11                                                                                                         

=== Evaluating checkpoint_step7940.pt | step=7940 | params=24,985,344 ===
[✓] step=7940 | avg_loss=27.1459 | batches=100 | tokens=30467 | ppl=6.16e+11                                                                                                         

=== Evaluating checkpoint_step8051.pt | step=8051 | params=24,985,344 ===
[✓] step=8051 | avg_loss=27.3890 | batches=100 | tokens=30467 | ppl=7.85e+11                                                                                                         

=== Evaluating checkpoint_step8163.pt | step=8163 | params=24,985,344 ===
[✓] step=8163 | avg_loss=26.7401 | batches=100 | tokens=30467 | ppl=4.10e+11                                                                                                         

=== Evaluating checkpoint_step8271.pt | step=8271 | params=24,985,344 ===
[✓] step=8271 | avg_loss=26.8204 | batches=100 | tokens=30467 | ppl=4.45e+11                                                                                                         

=== Evaluating checkpoint_step8376.pt | step=8376 | params=24,985,344 ===
[✓] step=8376 | avg_loss=27.2582 | batches=100 | tokens=30467 | ppl=6.89e+11                                                                                                         

=== Evaluating checkpoint_step8480.pt | step=8480 | params=24,985,344 ===
[✓] step=8480 | avg_loss=26.5912 | batches=100 | tokens=30467 | ppl=3.54e+11                                                                                                         

=== Evaluating checkpoint_step8584.pt | step=8584 | params=24,985,344 ===
[✓] step=8584 | avg_loss=26.0641 | batches=100 | tokens=30467 | ppl=2.09e+11                                                                                                         

=== Evaluating checkpoint_step8688.pt | step=8688 | params=24,985,344 ===
[✓] step=8688 | avg_loss=25.6500 | batches=100 | tokens=30467 | ppl=1.38e+11                                                                                                         

=== Evaluating checkpoint_step8792.pt | step=8792 | params=24,985,344 ===
[✓] step=8792 | avg_loss=25.3989 | batches=100 | tokens=30467 | ppl=1.07e+11                                                                                                         

=== Evaluating checkpoint_step8897.pt | step=8897 | params=24,985,344 ===
[✓] step=8897 | avg_loss=25.5804 | batches=100 | tokens=30467 | ppl=1.29e+11                                                                                                         

=== Evaluating checkpoint_step9002.pt | step=9002 | params=24,985,344 ===
[✓] step=9002 | avg_loss=25.1369 | batches=100 | tokens=30467 | ppl=8.26e+10                                                                                                         

=== Evaluating checkpoint_step9107.pt | step=9107 | params=24,985,344 ===
[✓] step=9107 | avg_loss=24.8902 | batches=100 | tokens=30467 | ppl=6.45e+10                                                                                                         

=== Evaluating checkpoint_step9211.pt | step=9211 | params=24,985,344 ===
[✓] step=9211 | avg_loss=24.5898 | batches=100 | tokens=30467 | ppl=4.78e+10                                                                                                         

=== Evaluating checkpoint_step9315.pt | step=9315 | params=24,985,344 ===
[✓] step=9315 | avg_loss=24.5984 | batches=100 | tokens=30467 | ppl=4.82e+10                                                                                                         

=== Evaluating checkpoint_step9419.pt | step=9419 | params=24,985,344 ===
[✓] step=9419 | avg_loss=24.4268 | batches=100 | tokens=30467 | ppl=4.06e+10                                                                                                         

=== Evaluating checkpoint_step9524.pt | step=9524 | params=24,985,344 ===
[✓] step=9524 | avg_loss=24.2118 | batches=100 | tokens=30467 | ppl=3.27e+10                                                                                                         

=== Evaluating checkpoint_step9628.pt | step=9628 | params=24,985,344 ===
[✓] step=9628 | avg_loss=24.0037 | batches=100 | tokens=30467 | ppl=2.66e+10                                                                                                         

=== Evaluating checkpoint_step9732.pt | step=9732 | params=24,985,344 ===
[✓] step=9732 | avg_loss=23.8319 | batches=100 | tokens=30467 | ppl=2.24e+10                                                                                                         

=== Evaluating checkpoint_step9837.pt | step=9837 | params=24,985,344 ===
[✓] step=9837 | avg_loss=23.7036 | batches=100 | tokens=30467 | ppl=1.97e+10                                                                                                         

=== Evaluating checkpoint_step9941.pt | step=9941 | params=24,985,344 ===
[✓] step=9941 | avg_loss=23.4834 | batches=100 | tokens=30467 | ppl=1.58e+10                                                                                                         

=== BEST CHECKPOINT ===




=== Evaluating checkpoint_step10046.pt | step=10046 | params=24,985,344 ===
[✓] step=10046 | avg_loss=23.3412 | batches=250 | tokens=304327 | ppl=1.37e+10                                                                                                       

=== Evaluating checkpoint_step10150.pt | step=10150 | params=24,985,344 ===
[✓] step=10150 | avg_loss=23.1320 | batches=250 | tokens=304327 | ppl=1.11e+10                                                                                                       

=== Evaluating checkpoint_step10255.pt | step=10255 | params=24,985,344 ===
[✓] step=10255 | avg_loss=22.9082 | batches=250 | tokens=304327 | ppl=8.89e+09           

=== Evaluating checkpoint_step10360.pt | step=10360 | params=24,985,344 ===
[✓] step=10360 | avg_loss=22.8278 | batches=250 | tokens=304327 | ppl=8.20e+09                                                                                                       

=== Evaluating checkpoint_step10464.pt | step=10464 | params=24,985,344 ===
[✓] step=10464 | avg_loss=22.7873 | batches=250 | tokens=304327 | ppl=7.88e+09                                                                                                       

=== Evaluating checkpoint_step10568.pt | step=10568 | params=24,985,344 ===
[✓] step=10568 | avg_loss=22.4649 | batches=250 | tokens=304327 | ppl=5.71e+09                                                                                                       

=== Evaluating checkpoint_step7828.pt | step=7828 | params=24,985,344 ===
[✓] step=7828 | avg_loss=26.3336 | batches=250 | tokens=304327 | ppl=2.73e+11                                                                                                        

=== Evaluating checkpoint_step7940.pt | step=7940 | params=24,985,344 ===
[✓] step=7940 | avg_loss=27.1659 | batches=250 | tokens=304327 | ppl=6.28e+11                                                                                                        

=== Evaluating checkpoint_step8051.pt | step=8051 | params=24,985,344 ===
[✓] step=8051 | avg_loss=27.3739 | batches=250 | tokens=304327 | ppl=7.73e+11                                                                                                        

=== Evaluating checkpoint_step8163.pt | step=8163 | params=24,985,344 ===
[✓] step=8163 | avg_loss=26.7530 | batches=250 | tokens=304327 | ppl=4.16e+11                                                                                                        

=== Evaluating checkpoint_step8271.pt | step=8271 | params=24,985,344 ===
[✓] step=8271 | avg_loss=26.8506 | batches=250 | tokens=304327 | ppl=4.58e+11       


=== Evaluating checkpoint_step8376.pt | step=8376 | params=24,985,344 ===
[✓] step=8376 | avg_loss=27.3532 | batches=250 | tokens=304327 | ppl=7.57e+11                                                                                                        

=== Evaluating checkpoint_step8480.pt | step=8480 | params=24,985,344 ===
[✓] step=8480 | avg_loss=26.6790 | batches=250 | tokens=304327 | ppl=3.86e+11                                                                                                        

=== Evaluating checkpoint_step8584.pt | step=8584 | params=24,985,344 ===
[✓] step=8584 | avg_loss=26.1255 | batches=250 | tokens=304327 | ppl=2.22e+11                                                                                                        

=== Evaluating checkpoint_step8688.pt | step=8688 | params=24,985,344 ===
[✓] step=8688 | avg_loss=25.6815 | batches=250 | tokens=304327 | ppl=1.42e+11                                                                                                        

=== Evaluating checkpoint_step8792.pt | step=8792 | params=24,985,344 ===
[✓] step=8792 | avg_loss=25.3946 | batches=250 | tokens=304327 | ppl=1.07e+11                                                                                                        

=== Evaluating checkpoint_step8897.pt | step=8897 | params=24,985,344 ===
[✓] step=8897 | avg_loss=25.5845 | batches=250 | tokens=304327 | ppl=1.29e+11                                                                                                        

=== Evaluating checkpoint_step9002.pt | step=9002 | params=24,985,344 ===
[✓] step=9002 | avg_loss=25.1395 | batches=250 | tokens=304327 | ppl=8.28e+10                                                                                                        

=== Evaluating checkpoint_step9107.pt | step=9107 | params=24,985,344 ===
[✓] step=9107 | avg_loss=24.8993 | batches=250 | tokens=304327 | ppl=6.51e+10                                                                                                        

=== Evaluating checkpoint_step9211.pt | step=9211 | params=24,985,344 ===
[✓] step=9211 | avg_loss=24.5940 | batches=250 | tokens=304327 | ppl=4.80e+10                                                                                                        

=== Evaluating checkpoint_step9315.pt | step=9315 | params=24,985,344 ===
[✓] step=9315 | avg_loss=24.5943 | batches=250 | tokens=304327 | ppl=4.80e+10                                                                                                        

=== Evaluating checkpoint_step9419.pt | step=9419 | params=24,985,344 ===
[✓] step=9419 | avg_loss=24.4075 | batches=250 | tokens=304327 | ppl=3.98e+10                                                                                                        

=== Evaluating checkpoint_step9524.pt | step=9524 | params=24,985,344 ===
[✓] step=9524 | avg_loss=24.1904 | batches=250 | tokens=304327 | ppl=3.20e+10                                                                                                        

=== Evaluating checkpoint_step9628.pt | step=9628 | params=24,985,344 ===
[✓] step=9628 | avg_loss=23.9862 | batches=250 | tokens=304327 | ppl=2.61e+10                                                                                                        

=== Evaluating checkpoint_step9732.pt | step=9732 | params=24,985,344 ===
[✓] step=9732 | avg_loss=23.8160 | batches=250 | tokens=304327 | ppl=2.20e+10    