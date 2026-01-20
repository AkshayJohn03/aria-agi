# tools/find_best_from_tb_and_copy.py
import os, glob, shutil, argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def find_min_scalar(runs_dir, tag="eval/cross_entropy"):
    best_val = float("inf"); best_step=None; best_file=None
    for root,dirs,files in os.walk(runs_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                path = os.path.join(root,f)
                ea = EventAccumulator(path)
                ea.Reload()
                if tag in ea.Tags().get("scalars", []):
                    scalars = ea.Scalars(tag)
                    for s in scalars:
                        if s.value < best_val:
                            best_val = s.value
                            best_step = int(s.step)
    return best_step, best_val

def find_checkpoint_for_step(ckpt_dir, step):
    files = glob.glob(os.path.join(ckpt_dir, "*step_*.pt")) + glob.glob(os.path.join(ckpt_dir,"*interrupt_step_*.pt"))
    if not files: return None
    def step_from_name(p):
        import re
        m=re.search(r"(\d{4,})", os.path.basename(p))
        return int(m.group(1)) if m else -1
    files = sorted(files, key=step_from_name)
    # pick nearest:
    files = sorted(files, key=lambda p: abs(step_from_name(p)-step))
    return files[0] if files else None

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="artifacts/zia_dense_runs/runs")
    p.add_argument("--ckpt_dir", default="artifacts/zia_dense_runs/checkpoints")
    p.add_argument("--out", default="recovered_best.pt")
    args=p.parse_args()
    step,val=find_min_scalar(args.runs_dir)
    print("Best tb step, val:", step, val)
    ckpt=find_checkpoint_for_step(args.ckpt_dir, step)
    print("Nearest checkpoint file:", ckpt)
    if ckpt:
        shutil.copyfile(ckpt, args.out)
        print("Copied to", args.out)
