import os, shutil

base_dir = "my_datasets/processed/arrow_cleaned_v1"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# --- 1. Flatten test folder if nested (test/test -> test/)
nested_test = os.path.join(test_dir, "test")
if os.path.exists(nested_test):
    for f in os.listdir(nested_test):
        src = os.path.join(nested_test, f)
        dst = os.path.join(test_dir, f)
        if os.path.isfile(src):
            shutil.move(src, dst)
    shutil.rmtree(nested_test)
    print("[✓] Flattened test/ folder.")

# --- 2. Ensure train exists
if not os.path.exists(train_dir):
    raise RuntimeError("❌ No train folder found. Run cleanup first.")

# --- 3. If test is empty, create a dummy validation split from train
if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
    os.makedirs(test_dir, exist_ok=True)
    print("[i] test/ was empty. You can split some data from train later for validation.")
else:
    print("[✓] test/ exists and is ready.")

print("[✓] Folder structure ready for `load_from_disk`.")
