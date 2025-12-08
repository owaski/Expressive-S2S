"""
Stress17K Pre-computed Features Dataset - Lightning-Compatible Version
Loads pre-extracted w2v-BERT features from disk for efficient training.
Adapted from LJSpeech precomputed dataset for stress-aware TTS training.
"""
import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List
import numpy as np
from pathlib import Path
from indextts.utils.front import TextNormalizer, TextTokenizer

class PrecomputedDataset(Dataset):
    """
    Lightning-Compatible Precomputed Dataset that loads pre-extracted features.
    Designed for TTS training with pre-computed features.
    """
    
    def __init__(
        self,
        features_root: str,
        metadata_path: str,
        split: str = "train",
        max_audio_length: float = 20.0,   # seconds (for filtering)
        max_text_length: int = 600,       # tokens (for filtering)
        model_dir: str = "checkpoints"
    ):
        self.features_root = features_root
        self.metadata_path = metadata_path
        self.split = split
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.model_dir = model_dir
        
        bpe_path = os.path.join(model_dir, "bpe.model")
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"BPE tokenizer not found at: {bpe_path}")
        
        self.normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer(bpe_path, self.normalizer)
        print(f"===== DATASET ===== Vocab file: {self.tokenizer.vocab_file}")
        print(f"===== DATASET ===== Vocab size: {self.tokenizer.vocab_size}")
        
        # Load metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        # Filter by split
        self.metadata = [item for item in all_metadata]
        
        # Determine features directory
        # self.features_dir = os.path.join(features_root, f"w2v-BERT_features_{split}")
        self.features_dir = os.path.join(features_root, split)
        
        # Check if features directory exists
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"Pre-extracted features not found at: {self.features_dir}")
        
        # Build list of valid samples (where features exist)
        self.valid_samples = []
        self.feature_files = []
        
        for item in self.metadata:
            file_id = item["audio_id"]
            feature_path = os.path.join(self.features_dir, f"{file_id}.pt")
            
            if os.path.exists(feature_path):
                self.valid_samples.append(item)
                self.feature_files.append(feature_path)
        
        if not self.feature_files:
            raise FileNotFoundError(f"No matching .pt feature files found in: {self.features_dir}")
        
        print(f"✅ Loaded {split} split with {len(self.feature_files)} pre-computed feature files")
        
    def _process_text(self, text: str):
        """Process text into tokens (same as original dataset)"""
        # Tokenize using BPE
        text_tokens = self.tokenizer.tokenize(text)
        
        # Truncate if too long
        if len(text_tokens) > self.max_text_length:
            text_tokens = text_tokens[:self.max_text_length]
        
        # Convert tokens to IDs
        text_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        text_tensor = torch.tensor(text_ids, dtype=torch.long)
        
        return text_tensor, text_tokens

    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load pre-extracted features from disk.
        Returns the same format as LJSpeech dataset but with stress information.
        """
        file_path = self.feature_files[idx]
        features = torch.load(file_path, map_location='cpu')
        
        # Re-tokenize text_with_stress on-the-fly to ensure consistency
        # The saved features already have text_tokens from text_with_stress
        # But we re-tokenize to be safe
        text_with_stress = features.get("text_with_stress", features["text"])
        text_tensor, text_tokens_list = self._process_text(text_with_stress)
        
        return {
            "file_id": features["file_id"],
            "audio_16k": features["audio_16k"],              # [samples] - raw waveform at 16kHz
            "text_tokens": text_tensor,                      # [text_length] - token IDs from text_with_stress
            "text": features["text"],                        # Original text without stress markers
            "text_with_stress": text_with_stress,            # Text with stress markers
            "input_features": features["input_features"],     # [time, 160] - w2v-BERT input features
            "attention_mask": features["attention_mask"],     # [time] - attention mask
            "semantic_emb": features["semantic_emb"],         # [time, 1024] - semantic embeddings
            "semantic_length": features["semantic_length"],   # int - actual sequence length
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for Stress17K pre-computed features DataLoader.
    Handles padding for both original data and pre-extracted features.
    """
    batch_size = len(batch)
    
    # Collect metadata
    file_ids = [item["file_id"] for item in batch]
    texts = [item["text"] for item in batch]
    texts_with_stress = [item["text_with_stress"] for item in batch]
    
    # Pad audio sequences (16kHz raw waveforms)
    audio_16k_list = [item["audio_16k"] for item in batch]
    audio_16k_lengths = torch.tensor([audio.shape[0] for audio in audio_16k_list], dtype=torch.long)
    audio_16k_padded = torch.nn.utils.rnn.pad_sequence(
        audio_16k_list, batch_first=True, padding_value=0.0
    )
    
    # Pad text tokens (from text_with_stress)
    text_tokens_list = [item["text_tokens"] for item in batch]
    text_tokens_lengths = torch.tensor([len(tokens) for tokens in text_tokens_list], dtype=torch.long)
    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(
        text_tokens_list, batch_first=True, padding_value=0
    )
    
    # Pad pre-extracted w2v-BERT features
    input_features_list = [item["input_features"] for item in batch]
    input_features_lengths = torch.tensor([feat.shape[0] for feat in input_features_list], dtype=torch.long)
    input_features_padded = torch.nn.utils.rnn.pad_sequence(
        input_features_list, batch_first=True, padding_value=0.0
    )
    
    # Pad attention masks
    attention_mask_list = [item["attention_mask"] for item in batch]
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list, batch_first=True, padding_value=False
    )
    
    # Pad semantic embeddings
    semantic_emb_list = [item["semantic_emb"] for item in batch]
    semantic_emb_padded = torch.nn.utils.rnn.pad_sequence(
        semantic_emb_list, batch_first=True, padding_value=0.0
    )
    
    # Semantic lengths (actual lengths from attention masks)
    semantic_lengths = torch.tensor([item["semantic_length"] for item in batch], dtype=torch.long)
    
    return {
        "file_ids": file_ids,
        "texts": texts,
        "texts_with_stress": texts_with_stress,
        
        # Original data (for compatibility)
        "text_tokens": text_tokens_padded,           # [batch, max_text_length] - padded
        "text_lengths": text_tokens_lengths,         # [batch] - actual text lengths
        "audio_16k": audio_16k_list,                 # List of tensors (keep original format)
        "audio_lengths": audio_16k_lengths,          # [batch] - audio sample lengths
        
        # Pre-extracted w2v-BERT features
        "input_features": input_features_padded,     # [batch, max_time, 160] - padded w2v-BERT inputs
        "attention_mask": attention_mask_padded,     # [batch, max_time] - padded attention masks
        "semantic_emb": semantic_emb_padded,         # [batch, max_time, 1024] - padded semantic embeddings
        "semantic_lengths": semantic_lengths,        # [batch] - actual semantic sequence lengths
        "input_features_lengths": input_features_lengths,  # [batch] - input feature lengths
    }


def create_precomputed_dataloader(
    features_roots: list,
    metadata_paths: list,
    batch_size: int = 8,
    num_workers: int = 4,
    split: str = "train",
    model_dir: str = "checkpoints",
    **dataset_kwargs
) -> DataLoader:
    """Create DataLoader for pre-computed Stress17K features"""

    # Support both Python lists and OmegaConf ListConfig by converting to list
    from omegaconf import ListConfig
    if isinstance(features_roots, ListConfig):
        features_roots = list(features_roots)
    if isinstance(metadata_paths, ListConfig):
        metadata_paths = list(metadata_paths)

    if (isinstance(features_roots, list) and isinstance(metadata_paths, list)):
        if len(features_roots) != len(metadata_paths):
            raise ValueError("features_roots and metadata_paths must have the same length")
        datasets = []
        for feature_root, metadata_path in zip(features_roots, metadata_paths):
            dataset = PrecomputedDataset(
                features_root=feature_root,
                metadata_path=metadata_path,
                split=split,
                model_dir=model_dir,
                **dataset_kwargs
            )
            datasets.append(dataset)
        dataset = torch.utils.data.ConcatDataset(datasets)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train")
    )
    
    return dataloader


if __name__ == "__main__":
    # Test the precomputed dataset and dataloader
    print("Testing Pre-computed Stress17K dataset...")
    
    try:
        dataset = PrecomputedDataset(
            split="train_full_train",
            max_audio_length=20.0,
            max_text_length=600,
            model_dir="checkpoints"
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test single sample
        sample = dataset[0]
        print("\nSample keys:", sample.keys())
        print("File ID:", sample["file_id"])
        print("Audio shape:", sample["audio_16k"].shape)
        print("Text tokens shape:", sample["text_tokens"].shape)
        print("Text tokens:", sample["text_tokens"])
        print("Input features shape:", sample["input_features"].shape)
        print("Attention mask shape:", sample["attention_mask"].shape)
        print("Semantic embeddings shape:", sample["semantic_emb"].shape)
        print("Semantic length:", sample["semantic_length"])
        print("Text (no stress):", sample["text"])
        print("Text (with stress):", sample["text_with_stress"])
        
        # Test dataloader
        print("\nTesting DataLoader...")
        dataloader = create_precomputed_dataloader(
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            split="train_full_train",
            model_dir="checkpoints",
            max_audio_length=20.0,
            max_text_length=600
        )
        
        batch = next(iter(dataloader))
        print("\nBatch keys:", batch.keys())
        print("Batch size:", len(batch["file_ids"]))
        print("Batch input features shape:", batch["input_features"].shape)
        print("Batch semantic embeddings shape:", batch["semantic_emb"].shape)
        print("Batch semantic lengths:", batch["semantic_lengths"])
        print("Batch text lengths:", batch["text_lengths"])
        print("Sample text (no stress):", batch["texts"][0])
        print("Sample text (with stress):", batch["texts_with_stress"][0])
        
        print("\n✅ Pre-computed Stress17K dataset test completed!")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please ensure features were extracted using extract_features_stress17k.py")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
