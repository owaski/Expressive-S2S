import json
import sys
from datasets import Dataset, DatasetDict, Audio
from pathlib import Path
import re
from tqdm import tqdm
import soundfile as sf
import numpy as np


def extract_stress_pattern(transcription):
    """
    Extract stress pattern from transcription with asterisk markers.
    
    Args:
        transcription: String with words marked with * for stress (e.g., "In *my* opinion")
    
    Returns:
        dict with 'binary', 'indices', 'words' keys
    """
    # Remove asterisks and split into words
    words = re.findall(r'\*?([^*\s]+)\*?', transcription)
    
    # Find stressed words (marked with asterisks)
    stressed_words = re.findall(r'\*([^*]+)\*', transcription)
    stressed_words_set = set(stressed_words)
    
    # Create binary pattern and indices
    binary = []
    indices = []
    stressed_list = []
    
    for i, word in enumerate(words):
        # Check if this word is stressed (case-insensitive)
        is_stressed = any(word.lower() == stressed.lower() for stressed in stressed_words_set)
        binary.append(1 if is_stressed else 0)
        if is_stressed:
            indices.append(i)
            stressed_list.append(word)
    
    return {
        'binary': binary,
        'indices': indices,
        'words': stressed_list
    }


def clean_transcription(transcription):
    """Remove asterisks from transcription to get clean text."""
    return re.sub(r'\*([^*]+)\*', r'\1', transcription)

def convert_to_dataset(
    input_file,
    audio_dir,
    split_name,
):
    # Read expresso.jsonl
    print(f"Reading metadata from {input_file}...")
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    print(f"Found {len(data)} entries in {input_file}")
    
    # Process each entry
    dataset_samples = []
    audio_dir = Path(audio_dir)
    
    for idx, entry in tqdm(enumerate(data), desc="Processing entries", total=len(data)):
        audio_file = audio_dir / entry["relative_audio_path"]
        
        if not audio_file.exists():
            print(f"Warning: Audio file {audio_file} not found, skipping entry {idx}")
            continue
        
        # Load audio
        try:
            audio_array, sampling_rate = sf.read(str(audio_file))
            # Ensure audio is in the right format (float32, normalized)
            if audio_array.dtype != np.float32:
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                else:
                    audio_array = audio_array.astype(np.float32)
        except Exception as e:
            print(f"Error loading audio {audio_file}: {e}, skipping entry {idx}")
            continue
        
        # Get transcription
        transcription = entry.get('transcription', '')
        
        # Extract stress pattern
        stress_pattern = extract_stress_pattern(transcription)
        
        # Clean transcription (remove asterisks)
        clean_text = clean_transcription(transcription)
        
        # Create sample in the format expected by inference_example.py
        sample = {
            'transcription_id': entry.get('id', idx),
            'transcription': clean_text,
            # 'audio': {
            #     'array': audio_array,
            #     'sampling_rate': sampling_rate,
            #     'path': entry.get('relative_audio_path', ''),
            # },
            'audio': str(audio_file),
            'stress_pattern': stress_pattern,
            # Keep original metadata for reference
            'original_transcription': transcription,
            'speaker_id': entry.get('speakerid', ''),
            'speaker_name': entry.get('name', ''),
            'emotion': entry.get('emotion', ''),
            'emphasis': entry.get('emphasis', False),
            'source': entry.get('source', ''),
            'relative_audio_path': entry.get('relative_audio_path', ''),
            'text_description': entry.get('text_description', ''),
            'intrinsic_tags': entry.get('intrinsic_tags', ''),
            'situational_tags': entry.get('situational_tags', ''),
            'basic_tags': entry.get('basic_tags', ''),
            'all_tags': entry.get('all_tags', ''),
        }
        
        dataset_samples.append(sample)
    
    print(f"Successfully processed {len(dataset_samples)} samples")
    
    
    return dataset_samples

def save_ds_to_disk(dataset_samples, output_dir, split_name):
    # Create HuggingFace dataset
    print("Creating HuggingFace dataset...")
    dataset = Dataset.from_list(dataset_samples)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Create DatasetDict with the specified split
    dataset_dict = DatasetDict({split_name: dataset})
    
    # Save dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving dataset to {output_dir}...")
    dataset_dict.save_to_disk(str(output_dir))
    
    # Also save as JSON for easy inspection
    json_output_path = output_dir / f'{split_name}_dataset_info.json'
    with open(json_output_path, 'w') as f:
        json.dump({
            'num_samples': len(dataset_samples),
            'split': split_name,
            'features': list(dataset.features.keys()) if hasattr(dataset, 'features') else []
        }, f, indent=2)
    
    print(f"Dataset saved successfully!")
    print(f"  - Dataset directory: {output_dir}")
    print(f"  - Split: {split_name}")
    print(f"  - Number of samples: {len(dataset_samples)}")
    print(f"\nTo use this dataset in inference_example.py, update the code to:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{output_dir}')")
    print(f"  sample = dataset['{split_name}'][0]")


if __name__ == "__main__":
    # split_name = "holdout"
    split_name = sys.argv[1]
    output_dir = "/data/group_data/li_lab/danqingw/datasets/paraspeechcap/preprocessed"

    # expresso
    input_file = f"/home/danqingw/workspace/cs11777/Expressive-S2S/data/paraspeechcaps/emotions/{split_name}.expresso.jsonl"
    audio_dir = "/data/group_data/li_lab/danqingw/datasets/paraspeechcap/expresso/"
    expresso_samples = convert_to_dataset(input_file, audio_dir, split_name)

    # ears
    input_file = f"/home/danqingw/workspace/cs11777/Expressive-S2S/data/paraspeechcaps/emotions/{split_name}.ears.jsonl"
    audio_dir = "/data/group_data/li_lab/danqingw/datasets/paraspeechcap/ears/"
    ears_samples = convert_to_dataset(input_file, audio_dir, split_name)
    
    dataset_samples = expresso_samples + ears_samples
    save_ds_to_disk(dataset_samples, output_dir, split_name)
    