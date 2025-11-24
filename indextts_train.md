
## UV venv
```bash
cd index-tts
uv sync
source Expressive-S2S/index-tts/.venv/bin/activate
cd .. # working under the root directory
```

## Data Processing
Step 1: 
Download stress17k, save audio files locally, and process stressed texts. After process, the audio data will be saved to data/stress17k/train_full, the metadata json file will be saved to data/stress17k_metadata/train_full_metadata.json, and the 
```bash
python index-tts/data_processing/stress17k_processing.py \
--output-dir data   
```


Step 2: Split the data to train and val split annotated in train_full_metadata.json files. Train split is annotated with "train_full_val" in "split" in data/stress17k_metadata/train_full_metadata.json.
```bash
python index-tts/data_processing/split_train_val.py \
--metadata_path data/stress17k_metadata/train_full_metadata.json
```

Step 3 Extract pre-computed features for training. The features will be stored under data/stress17k/w2v-BERT_features_train_full_train and data/stress17k/w2v-BERT_features_train_full_val
```bash
python index-tts/data_processing/extract_features_stress17k.py \
--metadata_path data/stress17k_metadata/train_full_metadata.json \
--model_dir index-tts/checkpoints \
--output_dir data/stress17k \
--split_name train_full_train

python index-tts/data_processing/extract_features_stress17k.py \
--metadata_path data/stress17k_metadata/train_full_metadata.json \
--model_dir index-tts/checkpoints \
--output_dir data/stress17k \
--split_name train_full_val
```

## Training:
Step 1: Extend the original text embedding from 12001 to 12003 while preserving the original model states
```bash
python index-tts/data_processing/migrate_gpt_checkpoint.py \
--old_checkpoint index-tts/checkpoints/gpt.pth \
--config index-tts/checkpoints/config.yaml \
--output_checkpoint index-tts/checkpoints/gpt_extended.pth \
--old_vocab_size 12000 \
--new_vocab_size 12002
```

Step 2:
```bash
python index-tts/src/train_stress17k.py \
--config index-tts/config/finetune_stress17k_config.yaml
```


## Inference
```bash
python /data/user_data/willw2/course_project_repo/Expressive-S2S/index-tts/src/eval/inference_eval.py --ckpt_path experiment/finetune_outputs_stress17k/checkpoints/stress17k-epoch=30-val_loss=2.5772.ckpt --config_path index-tts/checkpoints/config.yaml --original_ckpt_dir index-tts/checkpoints
```