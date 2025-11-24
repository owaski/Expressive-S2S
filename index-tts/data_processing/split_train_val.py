"""
Split train_full dataset into train_full_train and train_full_val splits
with a 4:1 ratio (80% train, 20% val)
"""
import json
import random
import argparse
from pathlib import Path


def split_train_val(
    metadata_path: str = "/data/user_data/willw2/data/stress17k_metadata_train_full_converted.json",
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
    
    # Filter train_full samples
    train_full_samples = [item for item in metadata if item.get("split") == "train_full"]
    other_samples = [item for item in metadata if item.get("split") != "train_full"]
    
    print(f"train_full samples: {len(train_full_samples)}")
    print(f"Other split samples: {len(other_samples)}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle train_full samples
    shuffled_samples = train_full_samples.copy()
    random.shuffle(shuffled_samples)
    
    # Calculate split point
    train_count = int(len(shuffled_samples) * train_ratio)
    
    # Split into train and val
    train_samples = shuffled_samples[:train_count]
    val_samples = shuffled_samples[train_count:]
    
    # Update split labels
    for sample in train_samples:
        sample["split"] = "train_full_train"
    
    for sample in val_samples:
        sample["split"] = "train_full_val"
    
    print(f"\nSplit results:")
    print(f"  train_full_train: {len(train_samples)} samples ({len(train_samples)/len(train_full_samples)*100:.1f}%)")
    print(f"  train_full_val: {len(val_samples)} samples ({len(val_samples)/len(train_full_samples)*100:.1f}%)")
    
    # Combine all samples back together
    updated_metadata = train_samples + val_samples + other_samples
    
    print(f"\nTotal samples in updated metadata: {len(updated_metadata)}")
    
    # Determine output path
    if output_path is None:
        output_path = metadata_path
    
    # Save updated metadata
    print(f"Saving updated metadata to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(updated_metadata, f, indent=2)
    
    print(f"✅ Successfully split train_full into train_full_train and train_full_val!")
    
    # Print summary statistics
    split_counts = {}
    for item in updated_metadata:
        split = item.get("split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
    
    print("\nFinal split distribution:")
    for split_name, count in sorted(split_counts.items()):
        print(f"  {split_name}: {count} samples")
    
    return updated_metadata


def verify_split(metadata_path: str):
    """Verify the split by checking sample counts"""
    print(f"Verifying splits in {metadata_path}...")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    split_counts = {}
    for item in metadata:
        split = item.get("split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
    
    print("\nSplit distribution:")
    for split_name, count in sorted(split_counts.items()):
        print(f"  {split_name}: {count} samples")
    
    # Check if train_full still exists
    if "train_full" in split_counts:
        print("\n⚠️  Warning: 'train_full' split still exists in metadata!")
    else:
        print("\n✅ 'train_full' split has been successfully replaced")
    
    # Check train/val ratio
    if "train_full_train" in split_counts and "train_full_val" in split_counts:
        train_count = split_counts["train_full_train"]
        val_count = split_counts["train_full_val"]
        total = train_count + val_count
        ratio = train_count / val_count if val_count > 0 else 0
        print(f"\nTrain/Val ratio: {ratio:.2f}:1 ({train_count/total*100:.1f}% train, {val_count/total*100:.1f}% val)")


def main():
    parser = argparse.ArgumentParser(description="Split train_full into train_full_train and train_full_val")
    parser.add_argument("--metadata_path", 
                       default="/data/user_data/willw2/data/stress17k_metadata_train_full_converted.json",
                       help="Path to metadata JSON file")
    parser.add_argument("--output_path", 
                       default=None,
                       help="Path to save updated metadata (defaults to input path)")
    parser.add_argument("--train_ratio", 
                       type=float, 
                       default=0.8,
                       help="Ratio of training samples (0.8 = 80%% train, 20%% val)")
    parser.add_argument("--seed", 
                       type=int, 
                       default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--verify_only", 
                       action="store_true",
                       help="Only verify existing split without modifying")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_split(args.metadata_path)
    else:
        split_train_val(
            metadata_path=args.metadata_path,
            output_path=args.output_path,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        
        # Verify the results
        output_path = args.output_path if args.output_path else args.metadata_path
        verify_split(output_path)


if __name__ == "__main__":
    main()
