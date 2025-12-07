import utmos
import argparse
import os
import math
import json
from pathlib import Path
from tqdm import tqdm

# ignore warnings from utmos
import warnings
warnings.filterwarnings("ignore")

def mos_directory(model, directory_path: str) -> dict[str, float]:
    paths = [f for f in Path(directory_path).glob("**/*.wav")]
    scores = {}
    for path in tqdm(paths, desc="Computing MOS scores"):
        try:
            scores[str(path)] = float(model.calculate_wav_file(path))
        except Exception as e:
            print(f"Error processing {str(path)}: {e}")
    mean = sum(scores.values()) / len(scores)
    ci_95 = 1.96 * (math.sqrt(sum((x - mean) ** 2 for x in scores.values()) / len(scores))) / math.sqrt(len(scores))
    return scores, mean, ci_95

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MOS scores for all WAV files in a directory.")
    parser.add_argument("--input-dir", type=str, help="Path to the directory containing WAV files.")
    parser.add_argument("--output-dir", type=str, help="Path to the directory to save output JSON files.", default=".")
    args = parser.parse_args()

    model = utmos.Score()

    scores, mean, ci_95 = mos_directory(model, args.input_dir)

    # output results as json
    # make output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "mos_scores.json"), "w") as f:
        json.dump(scores, f, indent=4)
    with open(os.path.join(args.output_dir, "mos_summary.json"), "w") as f:
        json.dump({"mean_mos": mean, "ci_95": ci_95}, f, indent=4)

    print(f"Mean MOS: {mean:.2f} ± {ci_95:.2f} (95% CI)")
