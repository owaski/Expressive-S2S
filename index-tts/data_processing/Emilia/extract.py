import tarfile
import json
import os
from pathlib import Path
from tqdm import tqdm
import glob
import random

def extract_and_organize_emilia(
    tar_pattern,
    output_base_dir="/data/user_data/willw2/data/EMILIA",
    max_samples_per_tar=None  # None = extract all, or set a limit like 1000
):
    """
    Extract audio files from Emilia tar files and organize them locally.
    Updates JSON metadata to point to the new audio file locations.
    
    Structure:
    emilia_dataset/
    ├── EN-B000000/
    │   ├── audio/
    │   │   ├── EN_B00000_S00000_W000000.mp3
    │   │   └── ...
    │   └── metadata/
    │       ├── EN_B00000_S00000_W000000.json
    │       └── ...
    ├── EN-B000001/
    │   ├── audio/
    │   └── metadata/
    └── ...
    """
    
    # Find all tar files
    tar_files = sorted(glob.glob(tar_pattern))
    print(f"Found {len(tar_files)} tar files to process")

    
    if not tar_files:
        print(f"No tar files found matching pattern: {tar_pattern}")
        return
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    total_samples = 0
    
    for tar_idx, tar_path in enumerate(tar_files, 1):
        tar_basename = Path(tar_path).stem  # e.g., "EN-B000000"
        print(f"\n[{tar_idx}/{len(tar_files)}] Processing {tar_basename}...")
        
        # Create subdirectories for this tar
        audio_dir = os.path.join(output_base_dir, tar_basename, "audio")
        metadata_dir = os.path.join(output_base_dir, tar_basename, "metadata")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                # Get all members
                members = tar.getmembers()
                json_members = [m for m in members if m.name.endswith('.json')]
                
                samples_to_extract = len(json_members)
                if max_samples_per_tar:
                    samples_to_extract = min(samples_to_extract, max_samples_per_tar)
                
                print(f"  Extracting {samples_to_extract} samples (out of {len(json_members)} total)...")
                
                # Process each sample
                for json_member in tqdm(json_members[:samples_to_extract], desc=f"  {tar_basename}"):
                    # Extract and read JSON metadata
                    json_filename = os.path.basename(json_member.name)
                    json_output_path = os.path.join(metadata_dir, json_filename)
                    
                    with tar.extractfile(json_member) as f_in:
                        metadata = json.load(f_in)
                    
                    # Extract corresponding MP3 audio
                    mp3_name = json_member.name.replace('.json', '.mp3')
                    try:
                        mp3_member = tar.getmember(mp3_name)
                        mp3_filename = os.path.basename(mp3_member.name)
                        mp3_output_path = os.path.join(audio_dir, mp3_filename)
                        
                        with tar.extractfile(mp3_member) as f_in:
                            with open(mp3_output_path, 'wb') as f_out:
                                f_out.write(f_in.read())
                        
                        # Update the "wav" path in metadata to point to new location
                        if 'wav' in metadata:
                            # Store absolute path
                            metadata['wav'] = os.path.abspath(mp3_output_path)
                        
                        # Write updated JSON metadata
                        with open(json_output_path, 'w') as f_out:
                            json.dump(metadata, f_out, indent=2)
                        
                    except KeyError:
                        print(f"    Warning: MP3 file not found for {json_filename}")
                
                total_samples += samples_to_extract
                print(f"  ✓ Extracted {samples_to_extract} samples from {tar_basename}")
                
        except Exception as e:
            print(f"  ✗ Error processing {tar_basename}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"✓ Extraction complete!")
    print(f"  Total samples extracted: {total_samples}")
    print(f"  Output directory: {output_base_dir}")
    print(f"{'='*60}")
    
    # Create a summary file
    summary_path = os.path.join(output_base_dir, "extraction_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Emilia Dataset Extraction Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total tar files processed: {len(tar_files)}\n")
        f.write(f"Total samples extracted: {total_samples}\n")
        f.write(f"Output directory: {output_base_dir}\n")
    
    return output_base_dir


def create_unified_metadata(
    output_base_dir="/data/user_data/willw2/data/EMILIA",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    """
    Collect all JSON metadata files and create a unified metadata file with train/val/test splits.
    
    Args:
        output_base_dir: Base directory containing extracted data
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.2)
        test_ratio: Proportion of data for testing (default: 0.1)
        seed: Random seed for reproducibility
    """
    
    print(f"\n{'='*60}")
    print("Creating unified metadata file with train/val/test splits...")
    print(f"{'='*60}")
    
    # Validate split ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    
    # Find all JSON metadata files
    metadata_pattern = os.path.join(output_base_dir, "*/metadata/*.json")
    json_files = sorted(glob.glob(metadata_pattern))
    
    print(f"Found {len(json_files)} metadata files")
    
    if not json_files:
        print("No metadata files found!")
        return
    
    # Load all metadata
    all_metadata = []
    print("Loading metadata files...")
    for json_path in tqdm(json_files):
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                # Add the original metadata file path for reference
                metadata['metadata_path'] = json_path
                all_metadata.append(metadata)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
    
    print(f"Successfully loaded {len(all_metadata)} metadata entries")
    
    # Shuffle data with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(all_metadata)
    
    # Calculate split indices
    total_samples = len(all_metadata)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    # Assign splits
    print(f"\nAssigning splits:")
    print(f"  Train: {train_end} samples ({train_ratio*100:.1f}%)")
    print(f"  Val: {val_end - train_end} samples ({val_ratio*100:.1f}%)")
    print(f"  Test: {total_samples - val_end} samples ({test_ratio*100:.1f}%)")
    
    for i, metadata in enumerate(all_metadata):
        if i < train_end:
            metadata['split'] = 'train'
        elif i < val_end:
            metadata['split'] = 'val'
        else:
            metadata['split'] = 'test'
    
    # Save unified metadata file
    unified_metadata_path = os.path.join(output_base_dir, "metadata.json")
    print(f"\nSaving unified metadata to: {unified_metadata_path}")
    
    with open(unified_metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Also save split-specific metadata files
    print("Creating split-specific metadata files...")
    
    train_data = [m for m in all_metadata if m['split'] == 'train']
    val_data = [m for m in all_metadata if m['split'] == 'val']
    test_data = [m for m in all_metadata if m['split'] == 'test']
    
    with open(os.path.join(output_base_dir, "metadata_train.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_base_dir, "metadata_val.json"), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(os.path.join(output_base_dir, "metadata_test.json"), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Create a summary
    print(f"\n{'='*60}")
    print("✓ Unified metadata created successfully!")
    print(f"{'='*60}")
    print(f"Files created:")
    print(f"  - metadata.json (all data: {len(all_metadata)} samples)")
    print(f"  - metadata_train.json ({len(train_data)} samples)")
    print(f"  - metadata_val.json ({len(val_data)} samples)")
    print(f"  - metadata_test.json ({len(test_data)} samples)")
    print(f"\nLocation: {output_base_dir}")
    
    # Save split statistics
    stats_path = os.path.join(output_base_dir, "split_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("Dataset Split Statistics\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Train samples: {len(train_data)} ({len(train_data)/total_samples*100:.2f}%)\n")
        f.write(f"Val samples: {len(val_data)} ({len(val_data)/total_samples*100:.2f}%)\n")
        f.write(f"Test samples: {len(test_data)} ({len(test_data)/total_samples*100:.2f}%)\n")
        f.write(f"Random seed: {seed}\n")
    
    return unified_metadata_path


# Example usage with different scenarios:

if __name__ == "__main__":
    # Step 1: Extract data from tar files
    output_dir = extract_and_organize_emilia(
        tar_pattern="/data/user_data/willw2/cache/huggingface/hub/datasets--amphion--Emilia-Dataset/snapshots/d7f2f7340a6385696f3766c8049fa920a4707c07/Emilia/EN/EN-B00000*.tar",
        output_base_dir="/data/user_data/willw2/data/EMILIA"
    )
    
    # Step 2: Create unified metadata with train/val/test splits
    create_unified_metadata(
        output_base_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42  # For reproducibility
    )