from datasets import load_from_disk, DatasetDict

train = load_from_disk("datasets/processed/zia_ift_v3_clean/train")
val = load_from_disk("datasets/processed/zia_ift_v3_clean/val")

dd = DatasetDict({"train": train, "validation": val})
dd.save_to_disk("datasets/processed/zia_ift_v3_clean_combined")
print("✅ Combined dataset saved to datasets/processed/zia_ift_v3_clean_combined")
