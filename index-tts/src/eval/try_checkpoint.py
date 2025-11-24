"""
Script to load and inspect a checkpoint saved by IndexTTSLightningModulePrecomputed
"""
import torch
import sys
import os
from pathlib import Path
from pprint import pprint

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def load_and_check_checkpoint(checkpoint_path: str):
    """
    Load and inspect a PyTorch Lightning checkpoint
    
    Args:
        checkpoint_path: Path to the .ckpt file
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print("=" * 80)
    
    # Load checkpoint (with weights_only=False for PyTorch 2.6+)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print("✅ Checkpoint loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # 1. Basic checkpoint info
    print("=" * 80)
    print("1. BASIC CHECKPOINT INFO")
    print("=" * 80)
    print(f"Checkpoint keys: {list(checkpoint.keys())}\n")
    
    # 2. Epoch and global step
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"Global step: {checkpoint['global_step']}")
    
    # 3. Hyperparameters
    print("\n" + "=" * 80)
    print("2. HYPERPARAMETERS")
    print("=" * 80)
    if 'hyper_parameters' in checkpoint:
        print("Hyperparameters:")
        pprint(checkpoint['hyper_parameters'])
    
    # 4. State dict info
    print("\n" + "=" * 80)
    print("3. STATE DICT INFO")
    print("=" * 80)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Total parameters in state_dict: {len(state_dict)}")
        print(f"\nFirst 20 keys in state_dict:")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            tensor = state_dict[key]
            print(f"  {i+1}. {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        if len(state_dict) > 20:
            print(f"  ... and {len(state_dict) - 20} more keys")
    
    # 5. GPT-specific state dicts
    print("\n" + "=" * 80)
    print("4. GPT-SPECIFIC STATE DICTS")
    print("=" * 80)
    gpt_keys = [
        'gpt_state_dict',
        'text_embedding_state_dict',
        'mel_embedding_state_dict',
        'final_norm_state_dict',
        'text_head_state_dict',
        'mel_head_state_dict',
        'mel_pos_embedding_state_dict',
        'text_pos_embedding_state_dict',
        'mel_layer_pos_embedding_state_dict',
        'text_layer_pos_embedding_state_dict'
    ]
    
    for key in gpt_keys:
        if key in checkpoint:
            state = checkpoint[key]
            print(f"✅ {key}: {len(state)} parameters")
        else:
            print(f"❌ {key}: NOT FOUND")
    
    # 6. Optimizer state
    print("\n" + "=" * 80)
    print("5. OPTIMIZER STATE")
    print("=" * 80)
    if 'optimizer_states' in checkpoint:
        print(f"Number of optimizer states: {len(checkpoint['optimizer_states'])}")
        if checkpoint['optimizer_states']:
            opt_state = checkpoint['optimizer_states'][0]
            if 'state' in opt_state:
                print(f"Number of parameter states: {len(opt_state['state'])}")
            if 'param_groups' in opt_state:
                print(f"Number of parameter groups: {len(opt_state['param_groups'])}")
                if opt_state['param_groups']:
                    print(f"Learning rate: {opt_state['param_groups'][0].get('lr', 'N/A')}")
    
    # 7. LR scheduler state
    print("\n" + "=" * 80)
    print("6. LR SCHEDULER STATE")
    print("=" * 80)
    if 'lr_schedulers' in checkpoint:
        print(f"Number of LR schedulers: {len(checkpoint['lr_schedulers'])}")
        if checkpoint['lr_schedulers']:
            print("LR scheduler state keys:", list(checkpoint['lr_schedulers'][0].keys()))
    
    # 8. Callbacks state
    print("\n" + "=" * 80)
    print("7. CALLBACKS STATE")
    print("=" * 80)
    if 'callbacks' in checkpoint:
        print(f"Callbacks: {checkpoint['callbacks'].keys() if isinstance(checkpoint['callbacks'], dict) else 'Present'}")
    
    # 9. Detailed parameter count
    print("\n" + "=" * 80)
    print("8. PARAMETER COUNT BY COMPONENT")
    print("=" * 80)
    
    def count_params(state_dict):
        """Count total parameters in a state dict"""
        return sum(p.numel() for p in state_dict.values())
    
    if 'gpt_state_dict' in checkpoint:
        gpt_params = count_params(checkpoint['gpt_state_dict'])
        print(f"GPT transformer: {gpt_params:,} parameters")
    
    if 'text_embedding_state_dict' in checkpoint:
        text_emb_params = count_params(checkpoint['text_embedding_state_dict'])
        print(f"Text embedding: {text_emb_params:,} parameters")
    
    if 'mel_embedding_state_dict' in checkpoint:
        mel_emb_params = count_params(checkpoint['mel_embedding_state_dict'])
        print(f"Mel embedding: {mel_emb_params:,} parameters")
    
    if 'text_head_state_dict' in checkpoint:
        text_head_params = count_params(checkpoint['text_head_state_dict'])
        print(f"Text head: {text_head_params:,} parameters")
    
    if 'mel_head_state_dict' in checkpoint:
        mel_head_params = count_params(checkpoint['mel_head_state_dict'])
        print(f"Mel head: {mel_head_params:,} parameters")
    
    # Total trainable parameters
    if 'state_dict' in checkpoint:
        total_params = count_params(checkpoint['state_dict'])
        print(f"\nTotal parameters in checkpoint: {total_params:,}")
    
    # 10. Sample a few parameter values
    print("\n" + "=" * 80)
    print("9. SAMPLE PARAMETER VALUES")
    print("=" * 80)
    if 'gpt_state_dict' in checkpoint:
        gpt_state = checkpoint['gpt_state_dict']
        sample_keys = list(gpt_state.keys())[:3]
        for key in sample_keys:
            tensor = gpt_state[key]
            print(f"\n{key}:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Mean: {tensor.float().mean().item():.6f}")
            print(f"  Std: {tensor.float().std().item():.6f}")
            print(f"  Min: {tensor.float().min().item():.6f}")
            print(f"  Max: {tensor.float().max().item():.6f}")
    
    print("\n" + "=" * 80)
    print("✅ CHECKPOINT INSPECTION COMPLETE")
    print("=" * 80)
    
    return checkpoint


if __name__ == "__main__":
    # Default checkpoint path
    checkpoint_path = "/data/user_data/willw2/expressive_s2st/index-tts/experiment/finetune_outputs_stress17k/checkpoints/stress17k-epoch=31-val_loss=2.3590.ckpt"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load and inspect checkpoint
    checkpoint = load_and_check_checkpoint(checkpoint_path)
