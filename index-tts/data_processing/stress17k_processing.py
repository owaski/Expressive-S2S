#!/usr/bin/env python3
"""
Stress-17K Dataset Downloader and Organizer

This script handles the complete workflow for the Stress-17K dataset:
1. Downloads the dataset from HuggingFace
2. Extracts all audio files to WAV format
3. Organizes files into a clean directory structure
4. Creates comprehensive metadata files

Author: GitHub Copilot
Date: 2025-11-13
"""

import pandas as pd
import soundfile as sf
import io
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import argparse
import logging
import re
import json
import numpy as np
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class Stress17KProcessor:
    """Complete processor for Stress-17K dataset"""
    
    def __init__(self, base_output_dir="/data/user_data/willw2/data"):
        self.base_output_dir = base_output_dir
        self.dataset_name = "slprl/Stress-17K-raw"
        
        # Directory structure
        self.raw_dir = os.path.join(base_output_dir, "stress17k_raw")
        self.audio_dir = os.path.join(base_output_dir, "stress17k")
        self.metadata_dir = os.path.join(base_output_dir, "stress17k_metadata")
        
        logger.info(f"Initialized Stress-17K processor")
        logger.info(f"Base output directory: {self.base_output_dir}")
    
    def download_dataset(self, force_redownload=False):
        """Download the Stress-17K dataset from HuggingFace"""
        logger.info("=" * 60)
        logger.info("STEP 1: DOWNLOADING STRESS-17K DATASET")
        logger.info("=" * 60)
        
        if os.path.exists(self.raw_dir) and not force_redownload:
            logger.info(f"✓ Dataset already exists at {self.raw_dir}")
            logger.info("  Use force_redownload=True to re-download")
            return True
        
        try:
            # Create directory
            os.makedirs(self.raw_dir, exist_ok=True)
            
            logger.info(f"📥 Downloading {self.dataset_name} to {self.raw_dir}")
            
            # Download using HuggingFace datasets
            ds = load_dataset(self.dataset_name)
            
            # Save as parquet files
            train_full_path = os.path.join(self.raw_dir, "data", "train_full-00000-of-00001.parquet")
            train_fine_path = os.path.join(self.raw_dir, "data", "train_fine-00000-of-00001.parquet")
            
            os.makedirs(os.path.dirname(train_full_path), exist_ok=True)
            
            # Convert to pandas and save
            logger.info("💾 Converting and saving train_full split...")
            df_full = ds['train_full'].to_pandas()
            df_full.to_parquet(train_full_path, index=False)
            
            logger.info("💾 Converting and saving train_fine split...")
            df_fine = ds['train_fine'].to_pandas()
            df_fine.to_parquet(train_fine_path, index=False)
            
            logger.info(f"✅ Dataset downloaded successfully!")
            logger.info(f"  • Train full: {len(df_full)} samples")
            logger.info(f"  • Train fine: {len(df_fine)} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading dataset: {e}")
            return False
    
    def extract_audio_split(self, parquet_path, output_dir, split_name, max_samples=None):
        """Extract audio from a single split to WAV files"""
        logger.info(f"🔄 Processing {split_name} split...")
        logger.info(f"📁 Input: {parquet_path}")
        logger.info(f"📁 Output: {output_dir}")
        
        # Load data
        df = pd.read_parquet(parquet_path)
        total_samples = len(df)
        n_samples = min(max_samples, total_samples) if max_samples else total_samples
        
        logger.info(f"📊 Processing {n_samples}/{total_samples} samples")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track metadata
        metadata_rows = []
        successful_extractions = 0
        failed_extractions = 0
        
        # Extract audio files
        for i in tqdm(range(n_samples), desc=f"Extracting {split_name} audio"):
            try:
                sample = df.iloc[i]
                audio_data = sample['audio']
                audio_bytes = audio_data['bytes']
                
                # Convert bytes to audio array
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                
                # Create clean filename
                audio_id = sample['audio_id']
                transcription = sample['transcription'][:30]
                safe_transcription = "".join(c for c in transcription if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_transcription = safe_transcription.replace(' ', '_')
                
                filename = f"{split_name}_{i:04d}_{audio_id}_{safe_transcription}.wav"
                filepath = os.path.join(output_dir, filename)
                
                # Save WAV file
                sf.write(filepath, audio_array, sr)
                
                # Collect metadata
                metadata_rows.append({
                    'filename': filename,
                    'audio_id': audio_id,
                    'index': i,
                    'transcription': sample['transcription'],
                    'description': sample['description'],
                    'intonation': re.sub(r'\*\*([^*]+)\*\*', r'<*>\1</*>', sample['intonation']),
                    'label': sample['label'],
                    'gt_stress_indices': sample['gt_stress_indices'],
                    'duration_seconds': len(audio_array) / sr,
                    'sample_rate': sr,
                    'audio_shape': str(audio_array.shape),
                    'whistress_transcription': str(sample['whistress_transcription']),
                    'predicted_stress_whistress': str(sample['predicted_stress_whistress']),
                    'split': split_name
                })
                
                successful_extractions += 1
                
            except Exception as e:
                logger.warning(f"❌ Error processing sample {i}: {e}")
                failed_extractions += 1
                continue
        
        # Save metadata
        if metadata_rows:
            metadata_path = os.path.join(self.metadata_dir, f"{split_name}_metadata.json")
            os.makedirs(self.metadata_dir, exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(metadata_rows, f, indent=2, default=convert_numpy)
            
            logger.info(f"📊 {split_name} extraction complete:")
            logger.info(f"  ✅ Success: {successful_extractions} files")
            logger.info(f"  ❌ Failed: {failed_extractions} files")
            logger.info(f"  📋 Metadata: {metadata_path}")
        
        return successful_extractions, failed_extractions, metadata_rows
    
    def extract_all_audio(self, max_samples_per_split=None):
        """Extract all audio files from the dataset"""
        logger.info("=" * 60)
        logger.info("STEP 2: EXTRACTING AUDIO TO WAV FILES")
        logger.info("=" * 60)
        
        # Check if raw data exists
        train_full_path = os.path.join(self.raw_dir, "data", "train_full-00000-of-00001.parquet")
        train_fine_path = os.path.join(self.raw_dir, "data", "train_fine-00000-of-00001.parquet")
        
        if not os.path.exists(train_full_path) or not os.path.exists(train_fine_path):
            logger.error("❌ Raw dataset files not found. Please run download_dataset() first.")
            return False
        
        # Extract train_full
        output_dir_full = os.path.join(self.audio_dir, "train_full")
        success_full, fail_full, metadata_full = self.extract_audio_split(
            train_full_path, output_dir_full, "train_full", max_samples_per_split
        )
        
        # Extract train_fine
        output_dir_fine = os.path.join(self.audio_dir, "train_fine")
        success_fine, fail_fine, metadata_fine = self.extract_audio_split(
            train_fine_path, output_dir_fine, "train_fine", max_samples_per_split
        )
        
        # Create combined metadata
        all_metadata = metadata_full + metadata_fine
        if all_metadata:
            combined_path = os.path.join(self.metadata_dir, "combined_metadata.json")
            with open(combined_path, 'w') as f:
                json.dump(all_metadata, f, indent=2, default=convert_numpy)
            logger.info(f"📋 Combined metadata saved: {combined_path}")
        
        # Summary
        total_success = success_full + success_fine
        total_fail = fail_full + fail_fine
        
        logger.info("=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Total WAV files created: {total_success}")
        logger.info(f"❌ Total failed extractions: {total_fail}")
        
        return total_success > 0
    
    def organize_files(self):
        """Organize files into clean directory structure"""
        logger.info("=" * 60)
        logger.info("STEP 3: ORGANIZING FILE STRUCTURE")
        logger.info("=" * 60)
        
        # Calculate file sizes
        if os.path.exists(self.audio_dir):
            total_audio_size = 0
            audio_file_count = 0
            
            for root, dirs, files in os.walk(self.audio_dir):
                for file in files:
                    if file.endswith('.wav'):
                        total_audio_size += os.path.getsize(os.path.join(root, file))
                        audio_file_count += 1
            
            logger.info(f"📊 Audio files organized:")
            logger.info(f"  • Total WAV files: {audio_file_count}")
            logger.info(f"  • Total size: {total_audio_size / (1024**2):.1f} MB")
            logger.info(f"  • Location: {self.audio_dir}")
        
        if os.path.exists(self.metadata_dir):
            metadata_files = [f for f in os.listdir(self.metadata_dir) if f.endswith('.json')]
            logger.info(f"📊 Metadata files organized:")
            logger.info(f"  • JSON files: {len(metadata_files)}")
            logger.info(f"  • Location: {self.metadata_dir}")
            for f in metadata_files:
                logger.info(f"    - {f}")
    
    
    def run_complete_pipeline(self, max_samples_per_split=None, force_redownload=False):
        """Run the complete pipeline"""
        logger.info("🚀 STARTING STRESS-17K COMPLETE PIPELINE")
        logger.info(f"Target directory: {self.base_output_dir}")
        
        # Step 1: Download
        if not self.download_dataset(force_redownload):
            logger.error("❌ Pipeline failed at download step")
            return False
        
        # Step 2: Extract audio
        if not self.extract_all_audio(max_samples_per_split):
            logger.error("❌ Pipeline failed at extraction step")
            return False
        
        # Step 3: Organize
        self.organize_files()
        
        
        logger.info("🎉 STRESS-17K PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("FINAL DIRECTORY STRUCTURE:")
        logger.info("=" * 60)
        
        # Show final structure
        for root, dirs, files in os.walk(self.base_output_dir):
            if 'stress17k' in root:
                level = root.replace(self.base_output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                logger.info(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    logger.info(f"{subindent}{file}")
                if len(files) > 5:
                    logger.info(f"{subindent}... and {len(files) - 5} more files")
        
        return True


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Download and organize Stress-17K dataset')
    parser.add_argument('--output-dir', default='/data/user_data/willw2/data',
                       help='Base output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per split (None for all)')
    parser.add_argument('--force-redownload', action='store_true',
                       help='Force re-download even if files exist')
    
    args = parser.parse_args()
    
    # Create processor and run pipeline
    processor = Stress17KProcessor(args.output_dir)
    success = processor.run_complete_pipeline(
        max_samples_per_split=args.max_samples,
        force_redownload=args.force_redownload
    )
    
    if success:
        print("\n🎉 SUCCESS! Stress-17K dataset is ready to use!")
        print(f"📁 Audio files: {processor.audio_dir}")
        print(f"📊 Metadata: {processor.metadata_dir}")
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())