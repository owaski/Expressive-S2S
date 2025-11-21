import torch
from datasets import load_dataset, load_from_disk
from whistress import WhiStressInferenceClient
from tqdm import tqdm
from pathlib import Path
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/group_data/li_lab/danqingw/datasets/paraspeechcap/preprocessed")
    parser.add_argument("--split_name", type=str, default="dev")
    parser.add_argument("--save_dir", type=str, default="/data/group_data/li_lab/danqingw/datasets/paraspeechcap/whistress_predictions")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--verbose", type=bool, default=False, help="Whether to print verbose output")
    return parser.parse_args()


def run_batch_inference(
    dataset,
    split_name="holdout",
    client=None,
    max_samples=None,
    verbose=False,
):
    """
    Run WhiStress inference over an entire split and return predictions.

    Args:
        dataset: HuggingFace dataset or dictionary of splits.
        split_name: Split key to iterate over (default: "holdout").
        client: Optional pre-instantiated WhiStressInferenceClient.
        max_samples: Optional cap on the number of samples processed.
        verbose: If True, prints running progress messages.

    Returns:
        List of dictionaries with ground truth and predicted outputs.
    """
    if split_name not in dataset:
        raise KeyError(f"Split '{split_name}' not found in dataset.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    whistress_client = client or WhiStressInferenceClient(device=device)

    results = []
    split = dataset[split_name]

    for idx, sample in tqdm(enumerate(split), desc="Running batch inference", total=len(split)):
        if max_samples is not None and idx >= max_samples:
            break

        pred_transcription, pred_stresses = whistress_client.predict(
            audio=sample["audio"],
            transcription=None,  # Using whisper transcription for inference from audio only.
            return_pairs=False,
        )

        result = {
            "sample_index": idx,
            "source": sample.get("source"),
            "speaker_id": sample.get("speaker_id"),
            "speaker_name": sample.get("speaker_name"),
            "emotion": sample.get("emotion"),
            "emphasis": sample.get("emphasis"),
            "text_description": sample.get("text_description"),
            "intrinsic_tags": sample.get("intrinsic_tags"),
            "situational_tags": sample.get("situational_tags"),
            "basic_tags": sample.get("basic_tags"),
            "all_tags": sample.get("all_tags"),
            "transcription_id": sample.get("transcription_id"),
            "ground_truth_transcription": sample.get("transcription"),
            "predicted_transcription": pred_transcription,
            "predicted_stress_pattern": pred_stresses,
        }
        results.append(result)

        if verbose:
            print(
                f"[{idx + 1}/{len(split)}] "
                f"transcription_id={result['transcription_id']} "
                f"predicted_stresses={pred_stresses}"
            )

    return results


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path
    dataset = load_from_disk(dataset_path)
    print(dataset)
    split_name = args.split_name
    save_dir = args.save_dir
    max_samples = args.max_samples
    verbose = args.verbose

    print("Loading WhiStress model for inference...")
    client = WhiStressInferenceClient(device="cuda" if torch.cuda.is_available() else "cpu")

    results = run_batch_inference(dataset, split_name=split_name, client=client, max_samples=max_samples, verbose=verbose)

    # if results:
    #     first = results[0]
    #     print(f'GT transcription: {first["ground_truth_transcription"]}')
    #     print(f'GT stressed words: {first["ground_truth_stress_pattern"]}')
    #     print(f'Predicted transcription: {first["predicted_transcription"]}')
    #     print(f'Predicted stressed words: {first["predicted_stress_pattern"]}')
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{split_name}_whistress_predictions.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"{len(results)} Results saved to {save_dir / f"{split_name}_whistress_predictions.jsonl"}")

