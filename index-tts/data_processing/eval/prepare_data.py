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


def prepare_expresso(expresso_dir, output_dir):
    print(f"Preparing Expresso dataset at {expresso_dir}")

    emotions = ['default', 'happy', 'sad']

    transcription_txt = os.path.join(expresso_dir, "read_transcriptions.txt")
    data = []
    with open(transcription_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            audio_path = os.path.join(expresso_dir, f"audio_48khz/read/{line[0].split('_')[0]}/{line[0].split('_')[1]}/base/{line[0]}.wav")

            text = convert_emphasis_to_tags(line[1])
            if not os.path.exists(audio_path):
                continue
            if line[0].split("_")[1] == "singing":
                continue
            sample = {}
            if line[0].split("_")[1] in emotions:
                sample = {
                    "audio_id": line[0],
                    "audio_path": audio_path,
                    "transcription": line[1],
                    "stressed_text": text,
                    "duration_sec": get_audio_duration(audio_path),
                    "emotion": line[0].split("_")[1],
                    'stressed': True if '<*>' in text else False,
                }
            else:
                sample = {
                    "audio_id": line[0],
                    "audio_path": audio_path,
                    "transcription": line[1],
                    "stressed_text": text,
                    "duration_sec": get_audio_duration(audio_path),
                    "emotion": line[0].split("_")[1],
                    'stressed': True if '<*>' in text else False
                }
            data.append(sample)
            print(sample)

    output_metadata_json = os.path.join(output_dir, "expresso_metadata.json")
    with open(output_metadata_json, 'w') as f:
        json.dump(data, f, indent=4)

    # print statistics of the current dataset
    summary = {
        "total_samples": len(data),
        "stressed_samples": len([d for d in data if d['stressed']]),
        "non_stressed_samples": len([d for d in data if not d['stressed']]),
        "stressed_duration_hours": sum([d['duration_sec'] for d in data if d['stressed']]) / 3600,
        "non_stressed_duration_hours": sum([d['duration_sec'] for d in data if not d['stressed']]) / 3600,
    }
    with open(os.path.join(output_dir, "expresso_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)

def prepare_stresstest(stresstest_dir, output_dir):
    print("Preparing Stresstest dataset...")
    print(stresstest_dir)

    df = pd.read_parquet(stresstest_dir)
    print(df.head())

def prepare_seedtts_testset(seedtts_dir, output_dir):
    print(f"Preparing SeedTTS test set at {seedtts_dir}")

    data = []

    en_metadata_file = os.path.join(seedtts_dir, "en", "meta.lst")
    with open(en_metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Splits the line into 5 parts
            parts = line.strip().split('|')
            print(len(parts))

            audio_id = parts[0]
            transcription = parts[3]
            prompt_audio_path = os.path.join(seedtts_dir, "en", parts[2])
            audio_path = os.path.join(seedtts_dir, "en", "wavs", parts[0] + ".wav")
            duration_sec = get_audio_duration(audio_path)
            
            print(f"Task: Generate '{transcription}' with audio from {audio_path}")
            data.append({
                "audio_id": audio_id,
                "audio_path": audio_path,
                "prompt_audio_path": prompt_audio_path,
                "transcription": transcription,
                "duration_sec": duration_sec,
            })
    output_metadata_json = os.path.join(output_dir, "seedtts_testset_metadata.json")
    with open(output_metadata_json, 'w') as f:
        json.dump(data, f, indent=4)

    # print statistics of the current dataset
    summary = {
        "total_samples": len(data),
        "total_duration_hours": sum([d['duration_sec'] for d in data]) / 3600,
    }
    with open(os.path.join(output_dir, "seedtts_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)

def prepare_LibriTTS_testset(libritts_dir, output_dir):
    print(f"Preparing LibriTTS test set at {libritts_dir}")

    data = []
    audio_paths = glob.glob(os.path.join(libritts_dir, "*", "*", "*", "*.wav"))
    for audio_path in audio_paths:
        audio_id = os.path.splitext(os.path.basename(audio_path))[0]
        duration_sec = get_audio_duration(audio_path)
        transcription_path = audio_path.replace('.wav', '.normalized.txt')
        with open(transcription_path, 'r', encoding='utf-8') as f:
            transcription = f.read().strip()
        data.append({
            "audio_id": audio_id,
            "audio_path": audio_path,
            "prompt_audio_path": audio_path,
            "transcription": transcription,
            "duration_sec": duration_sec,
        })
    output_metadata_json = os.path.join(output_dir, "libritts_testset_metadata.json")
    with open(output_metadata_json, 'w') as f:
        json.dump(data, f, indent=4)
    summary = {
        "total_samples": len(data),
        "total_duration_hours": sum([d['duration_sec'] for d in data]) / 3600,
    }
    with open(os.path.join(output_dir, "libritts_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)

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
    # if args.expresso_dir:
    #     prepare_expresso(args.expresso_dir, args.output_dir)
    
    # if args.stresstest_dir:
    #     prepare_stresstest(args.stresstest_dir, args.output_dir)

    # if args.seedtts_dir:
    #     prepare_seedtts_testset(args.seedtts_dir, args.output_dir)
    # if args.libri_tts_dir:
    #     prepare_LibriTTS_testset(args.libri_tts_dir, args.output_dir)
    if args.libri_tts_r_dir:
        prepare_libri_tts_R_testset(args.libri_tts_r_dir, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for evaluation")
    parser.add_argument("--expresso_dir", type=str, help="Path to the Expresso dataset directory")
    parser.add_argument("--stresstest_dir", type=str, help="Path to the Stresstest dataset directory")
    parser.add_argument("--seedtts_dir", type=str, help="Path to the SeedTTS dataset directory")
    parser.add_argument("--libri_tts_dir", type=str, help="Path to the LibriTTS dataset directory")
    parser.add_argument("--libri_tts_r_dir", type=str, help="Path to the LibriTTS-R dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for prepared data")
    args = parser.parse_args()

    prepare_dataset(args)
