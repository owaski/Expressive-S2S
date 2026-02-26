import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))
# Add WhiStress to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "WhiStress")))
import numpy as np
import librosa
import torch
import torchaudio
import torch.multiprocessing as mp
import torch.nn as nn
import argparse
import json
import re
import glob
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# from indextts.infer_v2 import IndexTTS2

from whistress import WhiStressInferenceClient
from funasr import AutoModel


def convert_emphasis_to_tags(text):
    """
    Convert *word* format to <*>word</*> format.
    
    Args:
        text: String with *emphasized* words
        
    Returns:
        String with <*>emphasized</*> words
    """
    return re.sub(r'\*([^*]+)\*', r'<*>\1</*>', text)

def get_stress_labels(sentence):
    """
    Extract binary stress labels from a sentence with asterisk-marked stressed words.
    
    Args:
        sentence: String with stressed words marked by asterisks, e.g., "Tell the *board*"
    
    Returns:
        list: Binary labels (0 or 1) for each word, where 1 indicates stress
    
    Examples:
        >>> get_stress_labels("Tell the board the quantum experiment is *done*")
        [0, 0, 0, 0, 0, 0, 0, 1]
        >>> get_stress_labels("Tell the board the *quantum experiment* is done")
        [0, 0, 0, 0, 1, 1, 0, 0]
    """
    # Remove asterisks to get clean text, but track which words were stressed
    stressed_words = set()
    parts = sentence.split('*')
    
    # Odd-indexed parts (1, 3, 5...) are between asterisks = stressed
    for i in range(1, len(parts), 2):
        # Get the words in this stressed section
        words = parts[i].strip().split()
        for word in words:
            # Strip punctuation for matching
            clean_word = word.strip('.,!?;:').lower()
            stressed_words.add(clean_word)
    
    # Now split the original sentence (without asterisks) into words
    clean_sentence = sentence.replace('*', '')
    words = clean_sentence.split()
    
    # Label each word
    labels = []
    for word in words:
        clean_word = word.strip('.,!?;:').lower()
        labels.append(1 if clean_word in stressed_words else 0)
    
    return labels


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

def extract_binary_stress_from_strong_marker(formatted_text):
    """
    Takes an HTML-formatted string with <strong> tags and returns a 
    list of binary values (1 for stressed, 0 for unstressed).
    """
    words = formatted_text.split()
    binary_list = []
    
    for word in words:
        # Check if the word contains the strong tags
        if "<strong>" in word and "</strong>" in word:
            binary_list.append(1)
        else:
            binary_list.append(0)
            
    return binary_list

def get_emo_embedding(model, fpath: str, emo_key):
    rec_result = model.generate(fpath, 
                                output_dir=None, 
                                granularity="utterance", 
                                extract_embedding=True)
    emotion_idx = rec_result[0]['labels'].index(emo_key)
    emotion_score = rec_result[0]['scores'][emotion_idx]
    return rec_result[0]['feats'], emotion_score

# def get_spk_embedding(model, fpath: str) -> np.ndarray:
#     wav = preprocess_wav(fpath)
#     embed = model.embed_utterance(wav)
#     return embed

def get_cos_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    # normalize embeddings
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2) 
    return np.inner(emb1, emb2)

def unified_worker_n_pass(rank, world_size, audio_dirs, stress_output_dir):
    # This prevents WhiStress and other libraries from leaking tensors to cuda:0
    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id) 
        device_str = f"cuda:{device_id}"
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")
    print(f"Worker {rank} strictly locked to device {device}")
    # WhiStress internally initializes a Whisper model; the set_device(device_id) 
    # call above ensures its internal LayerNorms land on the correct GPU.
    whistress_client = WhiStressInferenceClient(device=str(device))

    emo2vec_model = AutoModel(
        model="iic/emotion2vec_plus_large",
        hub="hf",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
    )

    my_audio_dirs = audio_dirs[rank::world_size]

    for audio_dir in tqdm(my_audio_dirs, desc=f"Worker {rank} - Stress"):
        try:
            # audio_id_folder = os.path.join(stress_output_dir, os.path.basename(sample).replace('.wav', ''))
            generated_audios = glob.glob(os.path.join(audio_dir, "*.wav"))
            print(f"Worker {rank}: Found {len(generated_audios)} generated audios for sample {audio_dir}")
            if not generated_audios:
                print(f"Worker {rank}: No generated audios found for sample {audio_dir}, skipping.")
                continue
            for audio_path in generated_audios:
                # load metadata
                with open(audio_path.replace('.wav', '.json'), 'r') as f:
                    metadata = json.load(f)
                
                # verify stress pattern using WhiStress
                audio_array, sr = librosa.load(audio_path)
                audio = {'array': audio_array, 'sampling_rate': sr}
                scored = whistress_client.predict(
                    audio=audio,
                    transcription=metadata['original_text'], 
                    return_pairs=True
                )
                pred_stress_pattern = [s[1] for s in scored]
                # gt_stress_pattern = get_stress_labels(metadata['stressed_text'])
                gt_stress_pattern = extract_binary_stress_from_strong_marker(metadata['stressed_text'])



                if metadata['emotion'] == "Happy" or metadata['emotion'] == 'happy':
                    emo_key = '开心/happy' # the key of emotion scores in FunASR's output dictionary 
                elif metadata['emotion'] == "Sad" or metadata['emotion'] == 'sad':
                    emo_key = '难过/sad' # the key of emotion scores in FunASR's output dictionary

                pred_emo_embed, pred_emo_score = get_emo_embedding(
                    emo2vec_model, 
                    audio_path,
                    emo_key
                )

                gt_emo_embed, _ = get_emo_embedding(
                    emo2vec_model,
                    metadata['speaker_prompt'],
                    emo_key
                )
                
                # result['audio_path'] = audio_path
                # result['stressed_transcription'] = sample['intonation']

                metadata['ground_truth_stress_pattern'] = gt_stress_pattern
                metadata['predicted_stress_pattern'] = pred_stress_pattern
                # print(f"the current audio_path is {audio_path}")
                # print(f"Worker {rank}: pred stress pattern: {pred_stress_pattern}")
                # print(f"Worker {rank}: gt stress pattern: {gt_stress_pattern}")
                metadata['f1_score'] = float(f1_score(gt_stress_pattern, pred_stress_pattern))
                metadata['precision_score'] = float(precision_score(gt_stress_pattern, pred_stress_pattern))
                metadata['recall_score'] = float(recall_score(gt_stress_pattern, pred_stress_pattern))
                metadata['emo_cos_sim'] = float(get_cos_sim(gt_emo_embed, pred_emo_embed))
                metadata['emo_score'] = float(pred_emo_score)

                stress_acc_output = audio_path.replace('.wav', '.json')
                with open(stress_acc_output, 'w') as f:
                    json.dump(metadata, f, indent=4)
        except Exception as e:
            print(f"Worker {rank}: Error processing stress sample {audio_dir}: {e}")

def eval_summary_n_pass(output_dir):
    audio_id_folders = glob.glob(os.path.join(output_dir, "audio*"))
    audio_paths = []
    all_f1 = []
    all_emo_sim = []
    correct_f1_emo_sim = []

    for audio_id_folder in tqdm(audio_id_folders):
        try:

            generated_audios = glob.glob(os.path.join(audio_id_folder, "*.wav"))
            current_f1 = {}
            for audio_path in generated_audios:
                json_path = audio_path.replace('.wav', '.json')

                if not os.path.exists(json_path):
                    print(f"Warning: JSON result not found for {audio_path}, skipping.")
                    continue
                with open(json_path, 'r') as f:
                    result = json.load(f)
                    print(json_path)
                    current_f1[audio_path] = [result['f1_score'], result['emo_cos_sim']]

            # find the audio with the highest f1 score, breaking ties with emo_sim
            if current_f1:
                # Find the maximum F1 score
                max_f1 = max(scores[0] for scores in current_f1.values())
                # Filter audios with the max F1 score
                candidates = {audio: scores for audio, scores in current_f1.items() if scores[0] == max_f1}
                # Among candidates, select the one with highest emo_sim
                best_audio = max(candidates, key=lambda x: candidates[x][1])
                best_f1 = current_f1[best_audio][0]
                best_emo_sim = current_f1[best_audio][1]
                all_f1.append(best_f1)
                all_emo_sim.append(best_emo_sim)    
                audio_paths.append(best_audio)

                if best_f1 == 1.0:
                    correct_f1_emo_sim.append(best_emo_sim)
        except Exception as e:
            print(f"Error processing audio folder {audio_id_folder}: {e}")

    mean_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    sd_f1 = (sum((x - mean_f1) ** 2 for x in all_f1) / len(all_f1)) ** 0.5 if all_f1 else 0.0
    mean_emo_sim = sum(all_emo_sim) / len(all_emo_sim) if all_emo_sim else 0.0
    sd_emo_sim = (sum((x - mean_emo_sim) ** 2 for x in all_emo_sim) / len(all_emo_sim)) ** 0.5 if all_emo_sim else 0.0
    print(f"SD of Emo Sim Scores: {sd_emo_sim:.4f}")
    print(f"Average Emo Sim Score across {len(all_emo_sim)} samples: {mean_emo_sim:.4f}")
    print(f"SD of F1 Scores: {sd_f1:.4f}")
    print(f"Average Best F1 Score across {len(all_f1)} samples: {mean_f1:.4f}")

    print(f"Average Emo Sim Score for perfectly stressed samples (F1=1.0): {sum(correct_f1_emo_sim) / len(correct_f1_emo_sim) if correct_f1_emo_sim else 0.0:.4f}")
    print(f"Number of perfectly stressed samples (F1=1.0): {len(correct_f1_emo_sim)} out of {len(all_f1)} total samples, ratio {len(correct_f1_emo_sim) / len(all_f1) if all_f1 else 0.0:.4f}")

    # save the best audio paths to a json file as the filtered dataset
    # each data has audio_path, f1_score, emo_sim_score, and stressed_transcription
    best_audios_data = []
    for audio_path in audio_paths:
        idx = audio_paths.index(audio_path)
        with open(audio_path.replace('.wav', '.json'), 'r') as f:
            result = json.load(f)
            stressed_text = result['stressed_text']
        
        if result['f1_score'] == 1.0:
            best_audios_data.append({
                # "domain": result['domain'],
                # "topic": result['topic'],
                # "sentence_type": result['sentence_type'],
                "intention": result['intention'],
                "emotion": result['emotion'],
                "audio_path": audio_path,
                "original_text": result['original_text'],
                "stressed_text": stressed_text, 
                "f1_score": all_f1[idx],
                "emo_sim_score": all_emo_sim[idx]
            })

    best_audios_output = os.path.join(output_dir, "verified_samples.json")
    with open(best_audios_output, 'w') as f:
        json.dump(best_audios_data, f, indent=4)
    summary_output = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_output, 'w') as f:
        f.write(f"SD of Emo Sim Scores: {sd_emo_sim:.4f}\n")
        f.write(f"Average Emo Sim Score across {len(all_emo_sim)} samples: {mean_emo_sim:.4f}\n")
        f.write(f"SD of F1 Scores: {sd_f1:.4f}\n")
        f.write(f"Average Best F1 Score across {len(all_f1)} samples: {mean_f1:.4f}\n")
        f.write(f"Average Emo Sim Score for perfectly stressed samples (F1=1.0): {sum(correct_f1_emo_sim) / len(correct_f1_emo_sim) if correct_f1_emo_sim else 0.0:.4f}\n")
        f.write(f"Number of perfectly stressed samples (F1=1.0): {len(correct_f1_emo_sim)} out of {len(all_f1)} total samples, ratio {len(correct_f1_emo_sim) / len(all_f1) if all_f1 else 0.0:.4f}\n")
    return all_f1, all_emo_sim, audio_paths


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

    stress_output_dir = args.output_dir

    audio_dirs = glob.glob(os.path.join(stress_output_dir, "audio*"))
    
    # Run evaluation with multiprocessing
    if num_workers > 1:
        mp.spawn(
            unified_worker_n_pass,
            args=(num_workers, audio_dirs, stress_output_dir),
            nprocs=num_workers,
            join=True
        )
    else:
        unified_worker_n_pass(0, 1, audio_dirs, stress_output_dir)
    return stress_output_dir


# def eval_summary(output_dir):
#     stress_results = glob.glob(os.path.join(output_dir, "*.json"))
#     all_f1 = []
#     for stress_result in stress_results:
#         with open(stress_result, 'r') as f:
#             result = json.load(f)
#             all_f1.append(result['f1_score'])

#     avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
#     standard_deviation = (sum((x - avg_f1) ** 2 for x in all_f1) / len(all_f1)) ** 0.5 if all_f1 else 0.0
#     print(f"Standard Deviation of Stress F1 Scores: {standard_deviation:.4f}")
#     print(f"Average Stress F1 Score across {len(all_f1)} samples: {avg_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS2 Inference with Stress-Aware Finetuned Model (Multiprocessing)")

    parser.add_argument(
        "--output_dir", 
        default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/audio_distribution_base_extra_word_emo_spk_random", 
        type=str, 
        help="Directory to save generated samples")
    
    args = parser.parse_args()

    # Full_evaluation(args)

    # eval_summary_n_pass(args.output_dir)

    # temp code to get a json file record wenetspeech prompt_audio_path
    base_dir = "/data/user_data/willw2/data/WenetSpeech4TTS/WenetSpeech4TTS_Premium_0"
    output_path = "/data/user_data/willw2/data/WenetSpeech4TTS/WenetSpeech4TTS_Premium_0/wenetspeech_premium_01_metadata.json"
    
    all_data = []
    audio_files = glob.glob(os.path.join(base_dir, "wavs", "*.wav"))
    for audio_file in tqdm(audio_files):
        # audio id
        audio_id = os.path.basename(audio_file).replace('.wav', '')
        print(audio_id)
        # audio duration
        info = torchaudio.info(audio_file)
        duration = info.num_frames / info.sample_rate
        # transcription
        txt_file = audio_file.replace('.wav', '.txt').replace("wavs", "txts")
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            audio_id, transcription = lines[0].strip().split('\t', 1)
            timestamps = json.loads(lines[1].strip())
        print(transcription)
        print(timestamps)

        all_data.append({
            "audio_id": audio_id,
            "audio_path": audio_file,
            "prompt_audio_path": audio_file, # in this case we use the same audio as the prompt for simplicity, but in practice you might want to use a different prompt audio
            "transcription": transcription,
            "duration_sec": duration,
            "timestamps": timestamps
        })
    
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)