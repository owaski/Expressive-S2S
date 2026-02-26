import gc
import sys
import os
from unittest import result
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))
# Add WhiStress to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "WhiStress")))
import numpy as np
import librosa
import torch
import torchaudio
import torch.multiprocessing as mp
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import torch.nn as nn
from torchinfo import summary
import argparse
import json
import re
import glob
import random
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import jiwer

from indextts.infer_v2 import IndexTTS2
from utils import insert_stress_tokens_preserving_positions

from whistress import WhiStressInferenceClient



'''
This script will be used to run inference using the stress_aware IndexTTS model.

Use IndexTTS2 from indextts.infer_v2
Load the finetuned GPT checkpoint from /data/user_data/willw2/expressive_s2st/index-tts/experiment/finetune_outputs_stress17k/checkpoints
This model contains other training artifacts, so only load the relevant modules into IndexTTS2 (such as the gpt module, text embedding, etc.)
Utilized stress words marked with the <*> </*> tokens for inference testing.
'''

def convert_emphasis_to_tags(text):
    """
    Convert *word* format to <*>word</*> format.
    
    Args:
        text: String with *emphasized* words
        
    Returns:
        String with <*>emphasized</*> words
    """
    return re.sub(r'\*([^*]+)\*', r'<*>\1</*>', text)

def remove_stress_markers(text):
    return text.replace('*', "")

def convert_stress_labels_stresstest(text):
    """
    Convert stress labels from Stresstest format to IndexTTS2 format.
    
    Args:
        text: Original text with stress labels (e.g., "LEONARDO PAINTED A REMARKABLE FRESCO.")
        
    Returns:
        Text with stress control markers inserted (e.g., "<*>LEONARDO</*> PAINTED A ...")
    """
    # For this example, let's assume the stress labels are indicated by uppercase words.
    words = text.split()
    stress_positions = []
    
    for idx, word in enumerate(words):
        if word.isupper():  # Assuming uppercase words are stressed
            stress_positions.append((idx, idx))
    
    stressed_text_tokens = insert_stress_tokens_preserving_positions(text, stress_positions, indextts2.tokenizer)
    stressed_text = ' '.join(stressed_text_tokens)
    
    return stressed_text

def get_stresstest_text(json_path):
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    texts = []
    for item in data:
        text = item['intonation']

        # convert "*I did not* steal this car."" to "<*>I did not</*> steal this car."
        text = re.sub(r'\*([^*]+)\*', r'<*>\1</*>', text)
        texts.append(text)
        print(f"converted text: {text}")

    return texts

def get_emo_embedding(model, fpath: str):
    rec_result = model.generate(fpath, 
                                output_dir=None, 
                                granularity="utterance", 
                                extract_embedding=True)
    return rec_result[0]['feats']

def get_spk_embedding(model, fpath: str) -> np.ndarray:
    wav = preprocess_wav(fpath)
    embed = model.embed_utterance(wav)
    return embed

def get_cos_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    # normalize embeddings
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2) 
    return np.inner(emb1, emb2)

def unified_worker(rank, world_size, stress_data):
    """
    Unified worker function that initializes all models once and processes all evaluation types.
    Each worker handles a subset of all three datasets.
    """
    # 1. CRITICAL FIX: Set the default CUDA device for this process
    # This prevents WhiStress and other libraries from leaking tensors to cuda:0
    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id) 
        device_str = f"cuda:{device_id}"
        device = torch.device(device_str)
        pipeline_device = device_id
    else:
        device = torch.device("cpu")
        pipeline_device = -1

    print(f"Worker {rank} strictly locked to device {device}")
    
    print(f"Worker {rank}: Loading WhiStress model...")
    # WhiStress internally initializes a Whisper model; the set_device(device_id) 
    # call above ensures its internal LayerNorms land on the correct GPU.
    whistress_client = WhiStressInferenceClient(device=str(device))
    
    print(f"Worker {rank}: All models loaded. Starting evaluation...")
    
    # Process stress samples
    my_stress_samples = stress_data[rank::world_size]
    print(f"\nWorker {rank}: Processing {len(my_stress_samples)} stress samples...")
    for sample in tqdm(my_stress_samples, desc=f"Worker {rank} - Stress"):
        # try:
        audio_path = sample['audio_path']
        print(f"Worker {rank}: Processing stress sample {audio_path}")
        print(os.path.exists(audio_path))
        # raise NotImplementedError("Need to adapt to new WhiStress API")
        audio_array, sr = librosa.load(audio_path)
        audio = {'array': audio_array, 'sampling_rate': sr}
        
        scored = whistress_client.predict(
            audio=audio,
            transcription=remove_stress_markers(sample['stressed_text']), 
            return_pairs=True
        )
        pred_stress_pattern = [s[1] for s in scored]
        sample["predicted_stress_pattern"] = pred_stress_pattern
        sample["f1_score"] = f1_score(sample['stressed_pattern'], pred_stress_pattern)
        sample["precision_score"] = precision_score(sample['stressed_pattern'], pred_stress_pattern)
        sample["recall_score"] = recall_score(sample['stressed_pattern'], pred_stress_pattern)
        metadata_path = audio_path.replace('audio', 'metadata').replace('.wav', '.json')
        print(os.path.exists(metadata_path))
        with open(metadata_path, 'w') as f:
            json.dump(sample, f, indent=4)
        # except Exception as e:
        #     print(f"Worker {rank}: Error processing stress sample {sample['audio_path']}: {e}")
    print(f"\nWorker {rank}: Completed all evaluations!")


def load_metadata(stresstest_jsonl_path):
    """
    Load stress evaluation data.
    """

    data = []
    json_paths = glob.glob(os.path.join(stresstest_jsonl_path, "*.json"))
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            sample = json.load(f)
            data.append(sample)
    return data


def Full_evaluation(args):
    """
    Run full evaluation using multiprocessing. Each worker initializes all models once
    and processes subsets of all evaluation datasets.
    """
    # Determine number of workers
    num_workers = args.num_workers if hasattr(args, 'num_workers') and args.num_workers else None
    if num_workers is None:
        num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    print(f"Starting evaluation with {num_workers} workers...")

    stress_data = load_metadata(args.metadata_dir)
    # stress_output_dir = args.output_dir

    # Run evaluation with multiprocessing
    if num_workers > 1:
        mp.spawn(
            unified_worker,
            args=(num_workers, stress_data),
            nprocs=num_workers,
            join=True
        )
    else:
        unified_worker(0, 1, stress_data)
    return args.metadata_dir


def eval_summary(output_dir):
    stress_results = glob.glob(os.path.join(output_dir, "*.json"))
    all_f1 = []
    for stress_result in stress_results:
        with open(stress_result, 'r') as f:
            result = json.load(f)
            all_f1.append(result['f1_score'])
    avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    standard_deviation = (sum((x - avg_f1) ** 2 for x in all_f1) / len(all_f1)) ** 0.5 if all_f1 else 0.0
    print(f"Standard Deviation of Stress F1 Scores: {standard_deviation:.4f}")
    print(f"Average Stress F1 Score across {len(all_f1)} samples: {avg_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS2 Inference with Stress-Aware Finetuned Model (Multiprocessing)")
    parser.add_argument("--metadata_dir", default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/audio_gemini_emo_intention_aware", type=str, help="Directory to save generated samples")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: number of GPUs available)")
    args = parser.parse_args()

    Full_evaluation(args)

    eval_summary(args.metadata_dir)
