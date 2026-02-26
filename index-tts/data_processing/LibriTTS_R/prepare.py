import os
import sys
import torch
import argparse
import json
import pandas as pd
import re
from src.utils import get_audio_duration
import glob
from whisper_normalizer.english import EnglishTextNormalizer
english_normalizer = EnglishTextNormalizer()

def convert_emphasis_to_tags(text):
    """
    Convert *word* format to <*>word</*> format.
    
    Args:
        text: String with *emphasized* words
        
    Returns:
        String with <*>emphasized</*> words
    """
    return re.sub(r'\*([^*]+)\*', r'<*>\1</*>', text)


def prepare_libri_tts_R_testset(libritts_dir, output_dir):
    print(f"Preparing LibriTTS-R test set at {libritts_dir}")

    data = []
    audio_paths = glob.glob(os.path.join(libritts_dir, "*", "*", "*", "*.wav"))
    for audio_path in audio_paths:
        audio_id = os.path.basename(audio_path).replace('.wav', '')
        duration_sec = get_audio_duration(audio_path)
        transcription_path = audio_path.replace('.wav', '.normalized.txt')
        with open(transcription_path, 'r', encoding='utf-8') as f:
            transcription = f.read().strip()
        data.append({
            "audio_id": audio_id,
            "audio_path": audio_path,
            "prompt_audio_path": audio_path,
            "transcription": english_normalizer(transcription),
            "duration_sec": duration_sec,
        })
    output_metadata_json = os.path.join(output_dir, "libritts_r_testset_metadata.json")
    with open(output_metadata_json, 'w') as f:
        json.dump(data, f, indent=4)
    summary = {
        "total_samples": len(data),
        "total_duration_hours": sum([d['duration_sec'] for d in data]) / 3600,
    }
    with open(os.path.join(output_dir, "libritts_r_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)


def prepare_dataset(args):
    if args.libri_tts_r_dir:
        prepare_libri_tts_R_testset(args.libri_tts_r_dir, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for evaluation")
    parser.add_argument("--libri_tts_r_dir", type=str, default="/data/user_data/willw2/data/LibriTTS_R", help="Path to the LibriTTS-R dataset directory")
    parser.add_argument("--output_dir", type=str, default="/data/user_data/willw2/data/LibriTTS_R", help="Path to the output directory for prepared data")
    args = parser.parse_args()

    prepare_dataset(args)
