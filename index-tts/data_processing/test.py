import json

# Path to your original metadata file
input_path = "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stress17k_metadata/train_full_metadata.json"

# Output file paths
train_path = "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stress17k_metadata/train_full_train_metadata.json"
val_path = "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stress17k_metadata/train_full_val_metadata.json"

# Read the original JSON file
with open(input_path, "r") as f:
    data = json.load(f)

# Split the data
train_data = [item for item in data if item.get("split") == "train_full_train"]
val_data = [item for item in data if item.get("split") == "train_full_val"]

# Write the split data to new files
with open(train_path, "w") as f:
    json.dump(train_data, f, indent=2)

with open(val_path, "w") as f:
    json.dump(val_data, f, indent=2)

print(f"Train: {len(train_data)} items\nVal: {len(val_data)} items")