import os
import sys
import json
import random
import argparse


def split_train_val(
    metadata_path: str,
    output_path: str = None,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    Split train_full samples into train_full_train and train_full_val
    
    Args:
        metadata_path: Path to the original metadata JSON file
        output_path: Path to save the updated metadata (defaults to same as input)
        train_ratio: Ratio of training samples (default 0.8 for 4:1 split)
        seed: Random seed for reproducibility
    """
    print(f"Loading metadata from {metadata_path}...")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Total samples: {len(metadata)}")
    
    # Split the samples
    random.seed(seed)

    shuffled_samples = metadata.copy()
    random.shuffle(shuffled_samples)

    train_count = int(len(shuffled_samples) * train_ratio)
    train_samples = shuffled_samples[:train_count]
    val_samples = shuffled_samples[train_count:]

    for sample in train_samples:
        sample["split"] = "train"
    for sample in val_samples:
        sample["split"] = "val"
    
    print(f"\nSplit results:")
    print(f"  train_full_train: {len(train_samples)} samples ({len(train_samples)/len(shuffled_samples)*100:.1f}%)")
    print(f"  train_full_val: {len(val_samples)} samples ({len(val_samples)/len(shuffled_samples)*100:.1f}%)")

    # combined all samples back together
    updated_metadata = train_samples + val_samples
    print(f"\nTotal samples in updated metadata: {len(updated_metadata)}")
    if output_path is None:
        output_path = metadata_path
    
    print(f"Saving updated metadata to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(updated_metadata, f, indent=4)
    
    # print summary statistics
    split_counts = {}
    for sample in updated_metadata:
        split = sample["split"]
        if split not in split_counts:
            split_counts[split] = 0
        split_counts[split] += 1
    for split, count in split_counts.items():
        print(f"  {split}: {count} samples ({count/len(updated_metadata)*100:.1f}%)")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split train_full samples into train_full_train and train_full_val")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to the original metadata JSON file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the updated metadata (defaults to same as input)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training samples (default 0.8 for 4:1 split)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    split_train_val(
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    
    
