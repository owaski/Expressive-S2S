#!/usr/bin/env python3

import pandas as pd
import soundfile as sf
import io
import os
import torch
import torchaudio
import shutil
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import argparse
import logging
import re
import json
import numpy as np
from collections import Counter
from transformers import SeamlessM4TFeatureExtractor
from indextts.utils.maskgct_utils import build_semantic_model
from indextts.utils.front import TextNormalizer, TextTokenizer
from src.utils import insert_stress_tokens_preserving_positions, get_stress_word_indices, remove_stress_control_markers

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

class Processor:
    """Complete processor for EMILIA dataset"""
    
    def __init__(self, base_output_dir, split, input_json_dir, extractor_path, device='cuda'):
        self.base_output_dir = base_output_dir
        self.max_audio_length = 20.0
        self.max_text_length = 600
        self.device = device
        
        # Directory structure
        self.split = split
        self.features_dir = os.path.join(base_output_dir, "precomputed_features", self.split)
        self.metadata_path = os.path.join(base_output_dir, f"EMILIA_metadata_{split}.json")
        self.precomputed_features_dir = os.path.join(base_output_dir, "precomputed_features", split)
        os.makedirs(self.precomputed_features_dir, exist_ok=True)

        self.input_json_dir = input_json_dir

        # Initialize feature extractor and semantic model
        logger.info(f"Initializing feature extractor on {device}...")
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        
        # Load semantic model for getting embeddings (same as lightning module)
        logger.info("Loading semantic model...")
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(extractor_path, "wav2vec2bert_stats.pt")
        )
        self.semantic_model = self.semantic_model.to(device).eval()
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        
        # Initialize text processing (same as LJSpeech dataset)
        bpe_path = os.path.join(extractor_path, "bpe.model") # use bpe_extended.model for stress control tokens
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"BPE tokenizer not found at: {bpe_path}")
        
        self.normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer(bpe_path, self.normalizer)
        
        logger.info(f"Initialized EMILIA processor")
        logger.info(f"Base output directory: {self.base_output_dir}")
    
    def add_stress_markers(self, text, stressed_words):
        """
        Add stress markers <*>word</*> to specified words in text.
        Handles multiple occurrences of the same word by marking them in order.
        
        Args:
            text: The original text
            stressed_words: List of words to stress (may contain punctuation and duplicates)
        
        Returns:
            Text with stress markers added
        """
        if not stressed_words:
            return text
        
        # Remove punctuation from stressed words (keep apostrophes for contractions)
        cleaned_words = [word.rstrip('.,!?;:"') for word in stressed_words]
        cleaned_words = [word for word in cleaned_words if word]  # Remove empty strings
        
        # Count how many times each word should be stressed
        stress_count = Counter(cleaned_words)
        
        # Track how many times we've marked each word
        marked_count = Counter()
        
        # Create a copy of the text to modify
        result = text
        offset = 0  # Track cumulative offset from adding markers
        
        # Find all word positions in the original text
        # Updated pattern to include apostrophes for contractions like "he's", "it's", etc.
        word_pattern = r"\b[\w']+\b"
        matches = list(re.finditer(word_pattern, text))
        
        for match in matches:
            word = match.group()
            word_lower = word.lower()
            
            # Check if this word needs to be stressed
            needs_stress = False
            for stress_word in stress_count:
                if word_lower == stress_word.lower():
                    # Check if we still need to mark more occurrences of this word
                    if marked_count[stress_word] < stress_count[stress_word]:
                        needs_stress = True
                        marked_count[stress_word] += 1
                        break
            
            if needs_stress:
                # Calculate position in the modified text
                start = match.start() + offset
                end = match.end() + offset
                
                # Add markers
                result = result[:start] + f"<*>{result[start:end]}</*>" + result[end:]
                
                # Update offset
                offset += len("<*></*>")
        return result
    
    def load_and_process_data(self):
        # load jsonl
        meta_data = []
        with open(self.input_json_dir) as f:
            data_ = json.load(f)
            data = [d for d in data_ if d['split'] == self.split]
        '''
            {
                "segment_id": 0,
                "audio_file": "/data/user_data/willw2/data/EMILIA/EN-B000000/diarized_audio/EN_B00000_S02900_W000061_part0.mp3",
                "start_sec": 0.0,
                "end_sec": 6.8,
                "duration_sec": 6.8,
                "text": "So now the question in front of us is whether the stability problems are anticipated",
                "stressed_words_transcription": "So now the question in front of us is whether the stability problems are <*>anticipated</*>",
                "all_transcriptions": [
                    "so now the question in front of us is whether the stability problems are <*>anticipated</*>.",
                    "so now the question in front of us is whether the stability problems are <*>anticipated</*>.",
                    "so now the question in front of us is whether the stability problems <*>are anticipated</*>.",
                    "so now the question in front of us is whether the stability problems are <*>anticipated</*>.",
                    "so now the question in front of us is whether the stability problems are <*>anticipated</*>."
                ],
                "word_frequencies": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    5
                ],
                "split": "train"
            },        
        '''
        count = 0

        for sample in data:
            audio_id = sample['audio_file']
            audio_path = sample['audio_file']
            
            # remove samples with no stress pattern
            if not sample['stressed_words_transcription']:
                count += 1
                continue

            if sample['duration_sec'] > self.max_audio_length:
                print(f"Skipping long audio file: {audio_path} ({sample['duration_sec']:.2f} seconds)")
                count += 1
                continue
            

            meta_data.append({
                'audio_id': audio_id,
                'audio_path': audio_path,
                'transcription': sample['text'],
                # 'predicted_stress_pattern': sample['predicted_stress_pattern'],
                "stressed_text": sample['stressed_words_transcription'],
            })

            # 
        print(f"Total skipped long audio files: {count}")

        return meta_data

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        """
        Extract semantic embeddings (same as lightning module)
        """
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat
    
    def _process_text(self, text):
        """Process text into tokens (same as LJSpeech dataset)"""
        # Tokenize using original BPE
        stress_positions = get_stress_word_indices(text)  # [(0, 0), (1, 1)]
        print(f"===== DEBUG ===== the stress_positions: {stress_positions}")
        normalized_text = remove_stress_control_markers(text)
        print(f"===== DEBUG ===== the normalized_text: {normalized_text}")
        tokens = self.tokenizer.tokenize(normalized_text)
        print(f"===== DEBUG ===== the tokens of normalized_text: {tokens}")

        subtokens = insert_stress_tokens_preserving_positions(normalized_text, stress_positions, self.tokenizer)
        print(f"===== DEBUG ===== the subtokens after inserting stress tokens: {subtokens}")
        text_tokens = subtokens
        
        # Truncate if too long
        if len(text_tokens) > self.max_text_length:
            text_tokens = text_tokens[:self.max_text_length]

        # Convert tokens to IDs
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        print(f"===== DEBUG ===== the id of text_with_stress tokens: {text_ids}")
        text_tensor = torch.tensor(text_ids, dtype=torch.long)
        
        return text_tensor, text_tokens
    
    def extract_single_sample(self, sample):
        audio_path = sample['audio_path']
        # Load and preprocess audio to 16kHz (same as LJSpeech)
        audio, sr = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Truncate if too long
        max_samples = int(self.max_audio_length * 16000)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
            
        audio_16k = audio.squeeze(0)  # [samples]
        # Process text with stress markers
        text_with_stress = sample["stressed_text"]
        text_tokens_with_stress, _ = self._process_text(text_with_stress)

        audio_list = [audio_16k.cpu()]
        # Extract features using SeamlessM4TFeatureExtractor
        extracted_input = self.feature_extractor(
            audio_list, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        input_features = extracted_input["input_features"].to(self.device)
        attention_mask = extracted_input["attention_mask"].to(self.device)
        semantic_emb = self.get_emb(input_features, attention_mask)
        # Calculate actual length
        actual_length = attention_mask.sum(dim=1).item()
    
        # Prepare data dictionary (same format as LJSpeech)
        # Note: text_tokens are created from text_with_stress for stress-aware training
        features = {
            "file_id": sample["audio_id"],
            "audio_16k": audio_16k,                                    # [samples] - raw waveform at 16kHz
            "text_tokens": text_tokens_with_stress,                                # [text_length] - token IDs from text_with_stress
            "text": sample["transcription"],                               # Original text without stress markers
            "text_with_stress": text_with_stress,                      # Text with stress markers (used for tokenization)

            # Extracted w2v-BERT features
            "input_features": input_features.cpu().squeeze(0),         # [time, 160] - w2v-BERT input features
            "attention_mask": attention_mask.cpu().squeeze(0),         # [time] - attention mask
            "semantic_emb": semantic_emb.cpu().squeeze(0),             # [time, 1024] - semantic embeddings
            "semantic_length": actual_length,                          # int - actual sequence length
        }
        return features
    
    def save_features(self, features, output_path):
        torch.save(features, output_path)

    def extract_features(self, data):
        """Extract features for all audio files in the metadata"""

        for sample in tqdm(data, desc="Extracting features"):
                file_id = sample["audio_id"]
                # Extract just the filename without extension from the full path
                file_basename = os.path.splitext(os.path.basename(file_id))[0]
                output_path = os.path.join(self.precomputed_features_dir, f"{file_basename}.pt")
                features = self.extract_single_sample(sample)
                # Save to disk
                self.save_features(features, output_path)
    
    
    def run_complete_pipeline(self, max_samples_per_split=None, force_redownload=False):
        """Run the complete pipeline"""
        logger.info("🚀 STARTING PARASPEECHCAP COMPLETE PIPELINE")
        logger.info(f"Target directory: {self.base_output_dir}")
        
        # Step 1: load and process paraspeechcap data
        data = self.load_and_process_data()

        # Step 2: save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

        
        # Step 2: Extract audio
        self.extract_features(data)



def main():
    """Main function with command line interface"""
    # base_output_dir, meta_dir, split
    parser = argparse.ArgumentParser(description='Process paraspeechcap dataset annotated data.')
    parser.add_argument('--base_output_dir', type=str, default='/data/user_data/willw2/course_project_repo/Expressive-S2S/data/paraspeechcaps',
                        help='Base output directory')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to process (default: train)')
    # parser.add_argument('--original_audio_dir', type=str, default='/data/group_data/li_lab/danqingw/datasets/paraspeechcap/',
    #                     help='Path to original audio directory')
    parser.add_argument('--input_json_path', type=str, default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/paraspeechcaps/emotion_w_stress/train_base.jsonl",
                        help='Path to input JSON file')
    parser.add_argument('--extractor_dir', type=str, default='/data/user_data/willw2/course_project_repo/Expressive-S2S/index-tts/checkpoints',
                        help='Path to feature extractor checkpoints')
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create processor and run pipeline
    processor = Processor(args.base_output_dir, args.split, args.input_json_path, args.extractor_dir, device=device)

    processor.run_complete_pipeline()

if __name__ == "__main__":
    main()