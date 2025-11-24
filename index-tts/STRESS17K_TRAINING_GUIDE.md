# Stress17K Fine-tuning Guide for IndexTTS 2.0

## Overview
This guide explains how to fine-tune the IndexTTS 2.0 text-to-semantic (GPT) model on the Stress17K dataset for stress-aware TTS.

## What's Been Done

### 1. Stress Token Integration
- **Token IDs**: `<*>` = 12003, `</*>` = 12004
- Modified `TextTokenizer` in `indextts/utils/front.py` to map stress tokens
- No retraining of tokenizer needed - using original `bpe.model`

### 2. Feature Extraction ✅
- **Script**: `src/data_processing/extract_features_stress17k.py`
- **Extracted**: 3520 training samples, 880 validation samples
- **Location**: `/data/user_data/willw2/data/stress17k/`
  - `w2v-BERT_features_train_full_train/` - 3520 .pt files
  - `w2v-BERT_features_train_full_val/` - 880 .pt files (needs extraction)

### 3. Dataset & Training Setup ✅
- **Dataset**: `src/stress17k_precomputed_dataset.py` - Loads pre-computed features
- **Config**: `config/finetune_stress17k_config.yaml` - Training configuration
- **Training Script**: `src/train_stress17k.py` - Main training script
- **Lightning Module**: `src/indextts_lightning_precomputed_clean.py` - Model wrapper

## Current Status

### Token ID Issue
The stress tokens are currently mapped to IDs **12003** and **12004** instead of the expected 12001/12002. This is because:
1. Original vocab: IDs 0-11999 (12000 tokens)
2. Reserved slot: ID 12000 (unused in original model)
3. Your mapping adds +1 and +2 on top of vocab_size (12000)
4. Result: 12001 and 12002, but features show 12003 and 12004

**Action needed**: The embedding layer needs to support IDs up to 12004, so size should be **12005** instead of 12001.

## Next Steps

### Step 1: Extract Validation Features
```bash
cd /data/user_data/willw2/expressive_s2st/index-tts
source .venv/bin/activate

python src/data_processing/extract_features_stress17k.py \
  --split_name train_full_val \
  --metadata_path /data/user_data/willw2/data/stress17k_metadata_train_full_converted.json \
  --model_dir /data/user_data/willw2/expressive_s2st/index-tts/checkpoints \
  --output_dir /data/user_data/willw2/data/stress17k
```

### Step 2: Update Model Embedding Size
The text embedding layer needs to be extended from 12001 to 12005 tokens.

**Option A: Extend embedding layer in config**
Update `config/finetune_stress17k_config.yaml`:
```yaml
gpt:
  number_text_tokens: 12005  # Was 12000, now support up to ID 12004
```

**Option B: Migrate checkpoint with extended embeddings**
If you have a checkpoint migration script, extend the text embedding from 12001→12005.

### Step 3: Test Training on Small Batch
```bash
cd /data/user_data/willw2/expressive_s2st/index-tts
source .venv/bin/activate

# Quick test with 1 GPU, small steps
python src/train_stress17k.py --config config/finetune_stress17k_config.yaml
```

### Step 4: Full Training
Modify `config/finetune_stress17k_config.yaml` if needed:
```yaml
hardware:
  gpus: 2  # Use 2 GPUs
  precision: "bf16"
  accumulate_grad_batches: 4

training:
  learning_rate: 1e-5
  warmup_steps: 500
  max_steps: 20000
  max_epochs: 100
```

Then run:
```bash
python src/train_stress17k.py --config config/finetune_stress17k_config.yaml
```

## Key Configuration Parameters

### Data Settings
```yaml
data:
  features_root: "/data/user_data/willw2/data/stress17k"
  metadata_path: "/data/user_data/willw2/data/stress17k_metadata_train_full_converted.json"
  batch_size: 16
  max_audio_length: 20.0  # seconds
  max_text_length: 600     # tokens
  train_split: "train_full_train"  # 3520 samples
  val_split: "train_full_val"      # 880 samples
```

### Training Settings
```yaml
training:
  learning_rate: 1e-5
  warmup_steps: 500
  max_steps: 20000
  gradient_clip_val: 1.0
```

### Hardware Settings
```yaml
hardware:
  gpus: 2
  precision: "bf16"
  accumulate_grad_batches: 4
```

## Files Created

### Dataset & Dataloader
- `src/stress17k_precomputed_dataset.py` - Dataset for loading pre-computed features
  - `Stress17KPrecomputedDataset` class
  - `collate_fn_stress17k_precomputed` function
  - `create_stress17k_precomputed_dataloader` function

### Configuration
- `config/finetune_stress17k_config.yaml` - Training config for Stress17K

### Training Script
- `src/train_stress17k.py` - Main training script

## Monitoring Training

### WandB Integration
Training logs to WandB by default:
```yaml
logging:
  use_wandb: true
  wandb_project: "indextts2-stress-tts"
  experiment_name: "indextts2_gpt_stress17k_finetune"
```

### Checkpoints
Checkpoints saved to:
```
finetune_outputs_stress17k/checkpoints/
  - stress17k-epoch=XX-val_loss=X.XXXX.ckpt
  - last.ckpt
```

## Troubleshooting

### Issue: "RuntimeError: Index out of range in embedding layer"
**Solution**: Embedding layer size is too small. Update `number_text_tokens` to 12005 in config.

### Issue: "FileNotFoundError: No .pt files found"
**Solution**: Run feature extraction script first for the missing split.

### Issue: "CUDA out of memory"
**Solutions**:
- Reduce `batch_size` from 16 to 8 or 4
- Increase `accumulate_grad_batches` to 8 or 16
- Reduce `max_audio_length` to filter long samples

### Issue: "Token ID 12003/12004 not found"
**Solution**: This is expected - features were extracted with these IDs. Just ensure embedding layer supports them.

## Dataset Statistics

- **Training samples**: 3520 (train_full_train split)
- **Validation samples**: 880 (train_full_val split)
- **Audio length**: ~2-4 seconds per sample
- **Text length**: ~10-25 tokens per sample
- **Stress markers**: Each sample has 1-2 stressed words marked with `<*>...</*>`

## Expected Training Time

With 2x GPUs, batch_size=16, accumulate_grad_batches=4:
- **Effective batch size**: 128
- **Steps per epoch**: ~27 steps (3520 / 128)
- **20K steps**: ~740 epochs
- **Estimated time**: ~6-12 hours (depending on GPU)

## Next Actions Summary

1. ✅ Features extracted for training (3520 samples)
2. ⏳ Extract features for validation (880 samples) - **DO THIS FIRST**
3. ⏳ Update `number_text_tokens` to 12005 in config
4. ⏳ Test training with small batch
5. ⏳ Run full training
6. ⏳ Evaluate stress control in generated speech

## Questions to Consider

1. **Embedding size**: Should we extend to 12005 or remap tokens to 12001/12002?
2. **Training duration**: Is 20K steps sufficient, or should we train longer?
3. **Learning rate**: Is 1e-5 appropriate for fine-tuning, or should we try 5e-6?
4. **Freezing**: Should we freeze more components (e.g., lower GPT layers)?
