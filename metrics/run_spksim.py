from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import argparse
import json
import os
import math
from tqdm import tqdm

def compute_embedding(encoder, fpath: str) -> np.ndarray:
    wav = preprocess_wav(fpath)

    embed = encoder.embed_utterance(wav)
    return embed

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return np.inner(emb1, emb2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute speaker similarity between ground truth and synthesized speech.")
    parser.add_argument("--data-file", type=str, required=True, help="Path to the JSON file containing preprocessed paths.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the similarity results.")
    args = parser.parse_args()

    encoder = VoiceEncoder()

    os.makedirs(args.output_dir, exist_ok=True)
    data = json.load(open(args.data_file, 'r'))
    results = {}
    for utt_id, paths in tqdm(data.items(), desc="Computing Speaker Similarity"):
        gt_path = paths["gt"]
        synth_path = paths["synth"]

        gt_embed = compute_embedding(encoder, gt_path)
        synth_embed = compute_embedding(encoder, synth_path)

        similarity = cosine_similarity(gt_embed, synth_embed)
        results[utt_id] = float(similarity)
    
    average_similarity = sum(results.values()) / len(results)
    ci_95 = 1.96 * (math.sqrt(sum((x - average_similarity) ** 2 for x in results.values()) / len(results))) / math.sqrt(len(results))

    # Save results
    with open(os.path.join(args.output_dir, "spksim_scores.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(args.output_dir, "spksim_summary.json"), "w") as f:
        json.dump({"average_spksim": average_similarity, "ci_95": ci_95}, f, indent=4)
    
    print(f"Average Speaker Similarity: {average_similarity:.4f} ± {ci_95:.4f} (95% CI)")
