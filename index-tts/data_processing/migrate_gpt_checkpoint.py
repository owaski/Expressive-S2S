"""
Vocabulary Extension Checkpoint Migration Utility

This script migrates an existing UnifiedVoice checkpoint to support extended vocabulary
while preserving learned embeddings for existing tokens.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pathlib import Path
import argparse
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint

def extend_embedding_layer(old_embedding: nn.Embedding, new_vocab_size: int, init_std: float = 0.02) -> nn.Embedding:
    """
    Extend an embedding layer to support larger vocabulary while preserving existing embeddings.
    
    Args:
        old_embedding: Existing embedding layer
        new_vocab_size: New vocabulary size (should be > old vocab size)
        init_std: Standard deviation for initializing new embeddings
    
    Returns:
        New embedding layer with extended vocabulary
    """
    old_vocab_size, embedding_dim = old_embedding.weight.shape
    
    if new_vocab_size <= old_vocab_size:
        raise ValueError(f"New vocab size ({new_vocab_size}) must be larger than old ({old_vocab_size})")
    
    print(f"Extending embedding: {old_vocab_size} → {new_vocab_size} (adding {new_vocab_size - old_vocab_size} tokens)")
    
    # Create new embedding layer
    new_embedding = nn.Embedding(new_vocab_size, embedding_dim)
    
    # Copy old weights
    new_embedding.weight.data[:old_vocab_size].copy_(old_embedding.weight.data)
    
    # Initialize new token embeddings with small random values
    nn.init.normal_(new_embedding.weight.data[old_vocab_size:], mean=0.0, std=init_std)
    
    return new_embedding

def extend_linear_layer(old_linear: nn.Linear, new_output_size: int, init_std: float = 0.02) -> nn.Linear:
    """
    Extend a linear layer to support larger output vocabulary.
    
    Args:
        old_linear: Existing linear layer
        new_output_size: New output size (should be > old output size)
        init_std: Standard deviation for initializing new weights
    
    Returns:
        New linear layer with extended output
    """
    old_output_size, input_size = old_linear.weight.shape
    
    if new_output_size <= old_output_size:
        raise ValueError(f"New output size ({new_output_size}) must be larger than old ({old_output_size})")
    
    print(f"Extending linear layer: {old_output_size} → {new_output_size} (adding {new_output_size - old_output_size} outputs)")
    
    # Create new linear layer
    new_linear = nn.Linear(input_size, new_output_size, bias=old_linear.bias is not None)
    
    # Copy old weights
    new_linear.weight.data[:old_output_size].copy_(old_linear.weight.data)
    
    # Initialize new output weights
    nn.init.normal_(new_linear.weight.data[old_output_size:], mean=0.0, std=init_std)
    
    # Copy old bias if it exists
    if old_linear.bias is not None:
        new_linear.bias.data[:old_output_size].copy_(old_linear.bias.data)
        # Initialize new bias entries to zero
        new_linear.bias.data[old_output_size:].zero_()
    
    return new_linear

def migrate_checkpoint(
    old_checkpoint_path: str,
    new_config_path: str,
    output_checkpoint_path: str,
    old_vocab_size: int,
    new_vocab_size: int
):
    """
    Migrate a checkpoint to support extended vocabulary.
    
    Args:
        old_checkpoint_path: Path to existing checkpoint
        new_config_path: Path to config with new vocabulary size
        output_checkpoint_path: Path to save migrated checkpoint
        old_vocab_size: Original vocabulary size (without +1 offset)
        new_vocab_size: New vocabulary size (without +1 offset)
    """
    
    print(f"Loading config from: {new_config_path}")
    cfg = OmegaConf.load(new_config_path)
    
    # Temporarily set to old vocab size to load checkpoint
    original_number_text_tokens = cfg.gpt.number_text_tokens
    cfg.gpt.number_text_tokens = old_vocab_size
    
    print(f"Creating model with old vocabulary size: {old_vocab_size}")
    old_model = UnifiedVoice(**cfg.gpt)
    
    print(f"Loading checkpoint from: {old_checkpoint_path}")
    load_checkpoint(old_model, old_checkpoint_path)
    
    # Now create new model with extended vocabulary
    cfg.gpt.number_text_tokens = new_vocab_size
    print(f"Creating model with new vocabulary size: {new_vocab_size}")
    new_model = UnifiedVoice(**cfg.gpt)
    
    # Copy all parameters except text_embedding and text_head
    print("Copying unchanged parameters...")
    old_state_dict = old_model.state_dict()
    new_state_dict = new_model.state_dict()
    
    for key in old_state_dict.keys():
        if key.startswith('text_embedding.') or key.startswith('text_head.'):
            continue  # Skip these - we'll handle them separately
        new_state_dict[key].copy_(old_state_dict[key])
    
    # Extend text_embedding
    print("Extending text_embedding...")
    new_model.text_embedding = extend_embedding_layer(
        old_model.text_embedding, 
        new_vocab_size + 1  # +1 for the types parameter
    )
    
    # Extend text_head  
    print("Extending text_head...")
    new_model.text_head = extend_linear_layer(
        old_model.text_head,
        new_vocab_size + 1  # +1 for the types parameter
    )
    
    # Save migrated model
    print(f"Saving migrated checkpoint to: {output_checkpoint_path}")
    
    # Save in the same format as the original checkpoint
    migrated_state_dict = new_model.state_dict()
    
    # Check if original checkpoint has 'model' key wrapper
    original_checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
    if 'model' in original_checkpoint:
        save_dict = {'model': migrated_state_dict}
    else:
        save_dict = migrated_state_dict
    
    torch.save(save_dict, output_checkpoint_path)
    
    print("✅ Migration completed successfully!")
    print(f"Old text_embedding: {old_vocab_size + 1} tokens")
    print(f"New text_embedding: {new_vocab_size + 1} tokens")
    print(f"Added {new_vocab_size - old_vocab_size} new token embeddings")
    
    return new_model

def main():
    parser = argparse.ArgumentParser(description="Migrate UnifiedVoice checkpoint for extended vocabulary")
    parser.add_argument("--old_checkpoint", required=True, help="Path to existing checkpoint")
    parser.add_argument("--config", required=True, help="Path to config with new vocabulary size") 
    parser.add_argument("--output_checkpoint", required=True, help="Path to save migrated checkpoint")
    parser.add_argument("--old_vocab_size", type=int, default=12000, help="Original vocabulary size")
    parser.add_argument("--new_vocab_size", type=int, default=12002, help="New vocabulary size")
    
    args = parser.parse_args()
    
    migrate_checkpoint(
        args.old_checkpoint,
        args.config, 
        args.output_checkpoint,
        args.old_vocab_size,
        args.new_vocab_size
    )

if __name__ == "__main__":
    main()