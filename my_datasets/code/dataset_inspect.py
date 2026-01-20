import os
import json
from datasets import load_from_disk
import xml.etree.ElementTree as ET

def human_size(nbytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}PB"

def count_jsonl_samples(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception as e:
        return f"Error: {e}"

def count_arrow_samples(directory):
    try:
        ds = load_from_disk(directory)
        return ds.num_rows
    except Exception as e:
        return f"Error: {e}"

def count_xml_samples(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        return len(list(root))
    except Exception as e:
        return f"Error: {e}"

def deep_inspect(base_dirs):
    print("\n🔍 Strong Dataset Inspection Starting...\n")
    results = []
    
    for base_dir in base_dirs:
        print(f"🗂 Scanning: {base_dir}")
        for root, dirs, files in os.walk(base_dir):
            # Ignore 'code' folder
            if "code" in dirs:
                dirs.remove("code")

            # Ignore __pycache__
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            
            for file in files:
                if file.endswith(('.py', '.md')):
                    continue

                fpath = os.path.join(root, file)
                rel_path = os.path.relpath(fpath, base_dir)
                fsize = human_size(os.path.getsize(fpath))
                sample_count = "N/A"
                dataset_type = "Unknown"

                if file.endswith(".jsonl"):
                    sample_count = count_jsonl_samples(fpath)
                    dataset_type = "JSONL"
                elif file.endswith(".arrow"):
                    sample_count = count_arrow_samples(root)
                    dataset_type = "Arrow"
                elif file.endswith(".xml"):
                    sample_count = count_xml_samples(fpath)
                    dataset_type = "XML"

                # Flagging small files (could be incomplete)
                flag = ""
                if isinstance(sample_count, int) and sample_count < 1000:
                    flag = "[⚠️ Very Small Sample Count]"

                print(f" - {rel_path} | Type: {dataset_type} | Size: {fsize} | Samples: {sample_count} {flag}")
                results.append({
                    "relative_path": rel_path,
                    "type": dataset_type,
                    "size": fsize,
                    "samples": sample_count,
                    "flag": flag.strip()
                })

    # Summary
    total_files = len(results)
    
    # Corrected logic to handle the size parsing
    unit_map = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4}
    total_size = 0
    for r in results:
        if r["size"] != "N/A":
            size_str = r["size"]
            unit = size_str[-2:]
            if unit in unit_map:
                value = float(size_str[:-2])
            else:
                unit = size_str[-1:]
                value = float(size_str[:-1])
            
            total_size += value * (1024 ** unit_map.get(unit, 0))

    print("\n✅ Dataset Inspection Complete")
    print(f"📊 Total files inspected: {total_files}")
    print(f"📊 Total size (approx.): {human_size(total_size)}")

    # Return results for optional export if needed later
    return results

if __name__ == "__main__":
    base_dirs = [
        r"D:\aria\aria_ai\aria_ai_assistant\my_datasets", # Corrected path
        r"F:\aria_ai\aria_ai_assistant\datasets",
        r"D:\aria\aria_ai\aria_ai_assistant\datasets"
    ]
    data_summary = deep_inspect(base_dirs)

    # Optional: Save to file
    import json
    with open("dataset_inspection_report.json", "w", encoding="utf-8") as f:
        json.dump(data_summary, f, indent=4)
    print("\n📂 Full dataset inspection saved to 'dataset_inspection_report.json'")