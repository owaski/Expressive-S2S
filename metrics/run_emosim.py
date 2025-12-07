from funasr import AutoModel
import numpy as np
import argparse
import json
import os
import math

model_id = "iic/emotion2vec_plus_large"

def compute_embedding(model, fpath: str):
    rec_result = model.generate(fpath, 
                                output_dir=None, 
                                granularity="utterance", 
                                extract_embedding=True)
    return rec_result[0]['feats']

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    # normalize embeddings
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2) 
    return np.inner(emb1, emb2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute emotion similarity between ground truth and synthesized speech.")
    parser.add_argument("--data-file", type=str, required=True, help="Path to the JSON file containing preprocessed paths.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the similarity results.")
    args = parser.parse_args()

    model = AutoModel(
        model=model_id,
        hub="hf",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
    )

    os.makedirs(args.output_dir, exist_ok=True)
    data = json.load(open(args.data_file, 'r'))
    results = {}
    for utt_id, paths in data.items():
        gt_path = paths["gt"]
        synth_path = paths["synth"]

        gt_embed = compute_embedding(model, gt_path)
        synth_embed = compute_embedding(model, synth_path)

        similarity = cosine_similarity(gt_embed, synth_embed)
        results[utt_id] = float(similarity)
    
    average_similarity = sum(results.values()) / len(results)
    ci_95 = 1.96 * (math.sqrt(sum((x - average_similarity) ** 2 for x in results.values()) / len(results))) / math.sqrt(len(results))

    # Save results
    with open(os.path.join(args.output_dir, "emosim_scores.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(args.output_dir, "emosim_summary.json"), "w") as f:
        json.dump({"average_emosim": average_similarity, "ci_95": ci_95}, f, indent=4)
    
    print(f"Average Emotion Similarity: {average_similarity:.4f} ± {ci_95:.4f} (95% CI)")
