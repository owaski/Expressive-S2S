"""
Offline Feature Extraction Script for Stress17K Dataset - IndexTTS Compatible
Extracts w2v-BERT features from Stress17K audio files and saves them to disk for efficient training.

This script is adapted from extract_features.py to work with the Stress17K dataset format.
Compatible with the existing IndexTTS pre-computed feature extraction workflow.
"""
import os
import torch
import torchaudio
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

from transformers import SeamlessM4TFeatureExtractor
from indextts.utils.maskgct_utils import build_semantic_model
from indextts.utils.front import TextNormalizer, TextTokenizer
from src.utils import insert_stress_tokens_preserving_positions, get_stress_word_indices, remove_stress_control_markers

class Stress17KFeatureExtractor:
    """Extract and save w2v-BERT features for Stress17K dataset offline"""
    
    def __init__(
        self,
        model_dir: str = "../checkpoints",
        output_dir: str = "/data/user_data/willw2/data/stress17k",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16
    ):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.device = device
        self.batch_size = batch_size
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize feature extractor and semantic model
        print(f"Initializing feature extractor on {device}...")
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        
        # Load semantic model for getting embeddings (same as lightning module)
        print("Loading semantic model...")
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(model_dir, "wav2vec2bert_stats.pt")
        )
        self.semantic_model = self.semantic_model.to(device).eval()
        self.semantic_mean = self.semantic_mean.to(device)
        self.semantic_std = self.semantic_std.to(device)
        
        # Initialize text processing (same as LJSpeech dataset)
        bpe_path = os.path.join(model_dir, "bpe.model") # use bpe_extended.model for stress control tokens
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"BPE tokenizer not found at: {bpe_path}")
        
        self.normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer(bpe_path, self.normalizer)
        print(f"===== DEBUG ===== the vocab file {self.tokenizer.vocab_file}")
        
        print(f"✅ Feature extractor initialized on {device}")
        
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
    
    def _process_text(self, text: str, max_text_length: int = 600):
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
        if len(text_tokens) > max_text_length:
            text_tokens = text_tokens[:max_text_length]
        
        # Convert tokens to IDs
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        print(f"===== DEBUG ===== the id of text_with_stress tokens: {text_ids}")
        text_tensor = torch.tensor(text_ids, dtype=torch.long)
        
        return text_tensor, text_tokens
    
    def extract_single_sample(self, sample_data: dict, max_audio_length: float = 20.0, max_text_length: int = 600):
        """
        Extract features for a single Stress17K sample
        Args:
            sample_data: Dictionary with Stress17K sample metadata
            max_audio_length: Maximum audio length in seconds
            max_text_length: Maximum text length in tokens
        Returns:
            dict with extracted features in the same format as LJSpeech
        """
        # Load audio
        audio_path = os.path.join(self.output_dir, sample_data["split"].replace("_train", "").replace("_val", ""), sample_data["filename"])
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
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
        max_samples = int(max_audio_length * 16000)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
            
        audio_16k = audio.squeeze(0)  # [samples]
        
        # Process text with stress markers
        text_with_stress = sample_data["intonation"]
        text_tokens_with_stress, _ = self._process_text(text_with_stress, max_text_length)
        print(f"===== DEBUG ===== the tokens of text_with_stress: {text_tokens_with_stress}")
        
        # Convert audio to list format for feature extractor
        audio_list = [audio_16k.cpu()]
        
        # Extract features using SeamlessM4TFeatureExtractor
        extracted_input = self.feature_extractor(
            audio_list, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        input_features = extracted_input["input_features"].to(self.device)
        attention_mask = extracted_input["attention_mask"].to(self.device)
        
        # Get semantic embeddings
        semantic_emb = self.get_emb(input_features, attention_mask)
        
        # Calculate actual length
        actual_length = attention_mask.sum(dim=1).item()
        
        # Prepare data dictionary (same format as LJSpeech)
        # Note: text_tokens are created from text_with_stress for stress-aware training
        features = {
            "file_id": sample_data["audio_id"],
            "audio_16k": audio_16k,                                    # [samples] - raw waveform at 16kHz
            "text_tokens": text_tokens_with_stress,                                # [text_length] - token IDs from text_with_stress
            "text": sample_data["transcription"],                               # Original text without stress markers
            "text_with_stress": text_with_stress,                      # Text with stress markers (used for tokenization)

            # Extracted w2v-BERT features
            "input_features": input_features.cpu().squeeze(0),         # [time, 160] - w2v-BERT input features
            "attention_mask": attention_mask.cpu().squeeze(0),         # [time] - attention mask
            "semantic_emb": semantic_emb.cpu().squeeze(0),             # [time, 1024] - semantic embeddings
            "semantic_length": actual_length,                          # int - actual sequence length
        }
        
        return features
    
    def save_features(self, features: dict, output_path: str):
        """Save features to a .pt file"""
        torch.save(features, output_path)
    
    def extract_from_metadata(
        self,
        metadata_path: str = "/data/user_data/willw2/data/stress17k_metadata_train_full_converted.json",
        max_samples: int = None,
        split_name: str = "train_full",
        **extract_kwargs
    ):
        """
        Extract features for Stress17K dataset from JSON metadata
        """
        print(f"Extracting features from {metadata_path}...")
        
        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded {len(metadata)} samples from metadata")
        
        # Filter by split if specified
        if split_name:
            metadata = [item for item in metadata if item.get("split") == split_name]
            print(f"Filtered to {len(metadata)} samples for split '{split_name}'")
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(metadata):
            metadata = metadata[:max_samples]
            print(f"Limiting to {max_samples} samples")
        
        # Create output directory with w2v-BERT prefix (same naming as LJSpeech)
        split_output_dir = os.path.join(self.output_dir, f"w2v-BERT_features_{split_name}")
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Process samples
        successful_extractions = 0
        print(f"Processing {len(metadata)} samples...")
        
        for idx, sample_data in enumerate(tqdm(metadata, desc=f"Extracting {split_name}")):
            file_id = sample_data["audio_id"]
            
            # Check if already processed
            output_path = os.path.join(split_output_dir, f"{file_id}.pt")
            if os.path.exists(output_path):
                successful_extractions += 1
                continue
            
            try:
                # Extract features
                features = self.extract_single_sample(sample_data, **extract_kwargs)
                
                # Save to disk
                self.save_features(features, output_path)
                successful_extractions += 1
                
            except Exception as e:
                print(f"Error processing sample {idx} (ID: {file_id}): {e}")
                continue
        
        print(f"✅ Feature extraction completed for {split_name} split!")
        print(f"Successfully processed {successful_extractions}/{len(metadata)} samples")
        print(f"Features saved to: {split_output_dir}")
        
        return successful_extractions


def verify_extraction(output_dir: str, split_name: str, num_samples: int = 5):
    """
    Verify extracted features by loading and checking a few samples
    """
    split_dir = os.path.join(output_dir, f"w2v-BERT_features_{split_name}")
    
    if not os.path.exists(split_dir):
        print(f"❌ Split directory not found: {split_dir}")
        return
        
    # Get list of extracted files
    pt_files = list(Path(split_dir).glob("*.pt"))
    
    if not pt_files:
        print(f"❌ No .pt files found in {split_dir}")
        return
    
    print(f"✅ Found {len(pt_files)} extracted feature files")
    
    # Check a few samples
    for i, pt_file in enumerate(pt_files[:num_samples]):
        try:
            features = torch.load(str(pt_file), map_location='cpu')
            
            print(f"\nSample {i+1}: {pt_file.name}")
            print(f"  File ID: {features['file_id']}")
            print(f"  Audio shape: {features['audio_16k'].shape}")
            print(f"  Text tokens shape: {features['text_tokens'].shape}")
            print(f"  Input features shape: {features['input_features'].shape}")
            print(f"  Attention mask shape: {features['attention_mask'].shape}")
            print(f"  Semantic embeddings shape: {features['semantic_emb'].shape}")
            print(f"  Semantic length: {features['semantic_length']}")
            print(f"  Text: {features['text'][:100]}...")
            if 'text_with_stress' in features:
                print(f"  Stress text: {features['text_with_stress']}...")
                print(f". Text tokens: {features['text_tokens']}...")


        except Exception as e:
            print(f"❌ Error loading {pt_file}: {e}")
    
    print(f"\n✅ Verification completed for {num_samples} samples")


def main():
    parser = argparse.ArgumentParser(description="Extract w2v-BERT features from Stress17K dataset")
    parser.add_argument("--metadata_path", 
                       default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stress17k_metadata/train_full_metadata.json",
                       help="Path to Stress17K metadata JSON file")
    parser.add_argument("--model_dir", default="index-tts/checkpoints", 
                       help="Path to model checkpoints")
    parser.add_argument("--output_dir", default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stress17k",
                       help="Output directory for extracted features")
    parser.add_argument("--split_name", default="train_full", 
                       help="Split name to process (should match 'split' field in metadata)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--max_audio_length", type=float, default=20.0,
                       help="Maximum audio length in seconds")
    parser.add_argument("--max_text_length", type=int, default=600,
                       help="Maximum text length in tokens")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for processing")
    parser.add_argument("--verify_only", action="store_true",
                       help="Only verify existing extracted features")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_extraction(args.output_dir, args.split_name)
        return
    
    # Initialize extractor
    extractor = Stress17KFeatureExtractor(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Extract features
    successful_count = extractor.extract_from_metadata(
        metadata_path=args.metadata_path,
        split_name=args.split_name,
        max_samples=args.max_samples,
        max_audio_length=args.max_audio_length,
        max_text_length=args.max_text_length
    )
    
    # Verify extraction
    if successful_count > 0:
        verify_extraction(args.output_dir, args.split_name)
    else:
        print("❌ No features were successfully extracted!")


if __name__ == "__main__":
    main()

    # test insertion function
    