# Suppress deprecation warnings
import warnings
warnings.filterwarnings('ignore')

# Fix for PyTorch 2.6+ weights_only issue with utmos/fairseq
import torch
# Monkey patch torch.load to default weights_only=False for compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import utmos
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def get_options():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate wav files using uT Mos model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--directory', '-d',
        type=str,
        default='Expressive-S2S/outputs/openai/expresso_emotion',
        help='Directory containing wav files to evaluate'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output JSON file path. If not specified, saves as utmos_scores.json in the input directory'
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        args.output = os.path.join(args.directory, 'utmos_scores.json')
    
    return args


def evaluate_directory(directory_path, output_json_path):
    """
    Evaluate all wav files in a directory and save scores to a JSON file.
    
    Args:
        directory_path: Path to directory containing wav files
        output_json_path: Path to output JSON file
    """
    # Initialize the model
    model = utmos.Score()
    
    # Get all wav files in the directory
    directory = Path(directory_path)
    wav_files = sorted(directory.glob('*.wav'))
    
    if not wav_files:
        print(f"No wav files found in {directory_path}")
        return
    
    print(f"Found {len(wav_files)} wav files")
    
    # Dictionary to store results
    results = {}
    
    # Evaluate each file with progress bar
    for wav_file in tqdm(wav_files, desc="Evaluating files", unit="file", ncols=100):
        try:
            score = model.calculate_wav_file(str(wav_file))
            results[wav_file.name] = {
                'filename': wav_file.name,
                'score': float(score)
            }
        except Exception as e:
            results[wav_file.name] = {
                'filename': wav_file.name,
                'score': None,
                'error': str(e)
            }
    
    # Save results to JSON
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_json_path}")
    
    # Print summary statistics
    valid_scores = [r['score'] for r in results.values() if r['score'] is not None]
    if valid_scores:
        print(f"\nSummary:")
        print(f"  Number of files evaluated: {len(results)}")
        print(f"  Number of successful evaluations: {len(valid_scores)}")
        print(f"  Average score: {sum(valid_scores) / len(valid_scores):.4f}")
        print(f"  Min score: {min(valid_scores):.4f}")
        print(f"  Max score: {max(valid_scores):.4f}")


if __name__ == "__main__":
    # Parse command-line arguments
    args = get_options()
    
    # Evaluate all files
    evaluate_directory(args.directory, args.output)