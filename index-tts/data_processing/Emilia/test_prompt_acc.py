import os
import sys
import google.generativeai as genai
import time
import json
import argparse
import glob
import multiprocessing
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

API_KEY="AIzaSyDhidZLY2k1SUQlYr5JJBH42hi4W0tmgYs"

examples = [
    {
        "path": "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stress17k/train_full/train_full_4386_f4d52db4-cccf-4da7-9996-65255a5dbf27_Join_us_now_to_discuss_the_eff.wav", 
        "ground_truth": """
        {
            "transcription": "join us now to discuss the <*>effects</*> of stress on health."
        }
        """
    },
    {
        "path": "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stress17k/train_full/train_full_0002_fcac4d7f-5780-4d07-a5f2-2d79ef97d10f_Leonardo_painted_a_remarkable.wav", 
        "ground_truth": """
        {
            "transcription": "leonardo painted a <*>remarkable</*> fresco."
        }
        """
    },
    {
        "path": "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stress17k/train_full/train_full_4331_c6c91722-cf66-4fd9-9c2e-25a8ff75244e_Relocate_the_urban_housing_pro.wav", 
        "ground_truth": """
        {
            "transcription": "relocate the urban housing projects <*>quickly</*>."
        }
        """
    },
    {
        "path": "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stress17k/train_full/train_full_4029_b395daca-ff83-464e-bc8f-7c9c7650fade_We_should_engage_the_audience.wav", 
        "ground_truth": """
        {
            "transcription": "we should engage the <*>audience</*> in our next play."
        }
        """
    }
]


sys_prompt = """
    this audio is performed by a voice actor given a transcript with indicators on which words are emphasized, the transcript might not have any emphasized words. find the possible emphasized words that could lead to this audio performance.
"""

# target_audio_path = "/data/user_data/willw2/data/EMILIA/EN-B000000/audio/EN_B00000_S00040_W000031.mp3"


def extract_stressed_words(text):
    """
    Extract stressed words from text with markers.
    Handles both <*>word</*> and *word* formats.
    Returns a set of stressed words (lowercased, no punctuation).
    """
    import re
    stressed = set()
    
    # Handle <*>word</*> format
    pattern1 = r'<\*>(.*?)</\*>'
    matches1 = re.findall(pattern1, text)
    
    # Handle *word* format
    pattern2 = r'\*([^*]+)\*'
    matches2 = re.findall(pattern2, text)
    
    for match in matches1 + matches2:
        # Clean and split multi-word expressions
        words = match.strip().replace('.', '').replace(',', '').replace('!', '').replace('?', '').lower().split()
        stressed.update(words)
    
    return stressed


def calculate_span_based_accuracy(predicted_text, ground_truth_text, original_transcription):
    """
    Calculate accuracy based on stressed word matching rather than position.
    This handles multi-word stressed expressions correctly.
    
    Returns: accuracy (as intersection over union), precision, recall, f1
    """
    pred_stressed = extract_stressed_words(predicted_text)
    gt_stressed = extract_stressed_words(ground_truth_text)
    
    # Get all words in the transcription
    all_words = set(original_transcription.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').split())
    
    if len(gt_stressed) == 0 and len(pred_stressed) == 0:
        # Both correctly identify no stress
        return 1.0, 1.0, 1.0, 1.0
    
    # Calculate metrics
    true_positives = len(pred_stressed & gt_stressed)
    false_positives = len(pred_stressed - gt_stressed)
    false_negatives = len(gt_stressed - pred_stressed)
    true_negatives = len(all_words - pred_stressed - gt_stressed)
    
    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    total = len(all_words)
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    
    # Precision: TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # F1: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1

def post_process_stress_markers(text):
    """
    Post-process stress markers in text:
    1. Move punctuation outside of stress markers
    2. Merge consecutive stressed words into single stress markers
    
    Args:
        text: String with stress markers in format <*>word</*>
        
    Returns:
        Processed string with cleaned stress markers
    
    Examples:
        >>> post_process_stress_markers("You chose to do <*>this?</*>")
        'You chose to do <*>this</*>?'
        
        >>> post_process_stress_markers("You chose to <*>do</*> <*>this</*>?")
        'You chose to <*>do this</*>?'
    """
    import re
    
    # Step 1: Move punctuation outside of stress markers
    # Find patterns like <*>word?</*> and convert to <*>word</*>?
    punctuation_pattern = r'<\*>(.*?)([.,!?;:]+)</\*>'
    text = re.sub(punctuation_pattern, r'<*>\1</*>\2', text)
    
    # Step 2: Merge consecutive stressed words
    # Find patterns like <*>word1</*> <*>word2</*> and convert to <*>word1 word2</*>
    # We need to do this iteratively until no more consecutive markers exist
    while True:
        consecutive_pattern = r'<\*>(.*?)</\*>\s+<\*>(.*?)</\*>'
        new_text = re.sub(consecutive_pattern, r'<*>\1 \2</*>', text)
        if new_text == text:
            break
        text = new_text
    
    return text


# --- HELPER FUNCTION TO UPLOAD ---
def upload_and_wait(path):
    print(f"Uploading {path}...")
    file_obj = genai.upload_file(path=path)
    
    # Wait for processing
    while file_obj.state.name == "PROCESSING":
        time.sleep(1)
        file_obj = genai.get_file(file_obj.name)
        
    if file_obj.state.name != "ACTIVE":
        raise ValueError(f"File {file_obj.name} failed to process.")
    return file_obj


def cal_acc_single(target_file, num_runs=5):
    # Debugging: Check the type and content of target_file
    print(f"Processing {target_file}")
    print(f"Type: {type(target_file)}")

    if not isinstance(target_file, dict):
        raise ValueError(f"Expected a dictionary, got {type(target_file)}: {target_file}")

    try:
        # init
        genai.configure(api_key=API_KEY)

        # build prompt
        prompt_prefix = []
        prompt_prefix.append(sys_prompt)
        ICL_examples = []
        for ex in examples:
            example_file = upload_and_wait(ex["path"])
            ICL_examples.append(example_file)

            prompt_prefix.append("Input Audio:")
            prompt_prefix.append(example_file)
            prompt_prefix.append("Correct Output:")
            prompt_prefix.append(ex["ground_truth"])
        prompt_prefix.append("Now, analyze this audio file following the same format:")
        prompt_prefix.append("The ground truth transcription is given for reference only, do not rely on it to find stressed words.")

        model = genai.GenerativeModel(
            "gemini-3-pro-preview",
            generation_config={"response_mime_type": "application/json"}
        )

        target_file_obj = None

        # upload target audio once (no need to upload multiple times)
        target_file_obj = upload_and_wait(target_file["audio_path"])
        # prepare prompt
        current_prompt = prompt_prefix.copy()

        current_prompt.append(f'\nTranscription: "{target_file["transcription"]}"')
        current_prompt.append(target_file_obj)

        # Run inference multiple times and collect results
        all_stressed_words_sets = []  # Store sets of stressed words from each run
        all_transcriptions = []
        
        for run_idx in range(num_runs):
            print(f"  Run {run_idx + 1}/{num_runs} for {target_file['audio_path']}")
            
            # inference
            response = model.generate_content(current_prompt)

            # post-process
            json_data = json.loads(response.text)
            # check if that is a list or not 
            if isinstance(json_data, list):
                json_data = json_data[0]
            all_transcriptions.append(json_data['transcription'])
            
            # Extract stressed words (handles multi-word expressions)
            stressed_words = extract_stressed_words(json_data['transcription'])
            all_stressed_words_sets.append(stressed_words)
            
            # Add a small delay between runs to avoid rate limiting
            if run_idx < num_runs - 1:
                time.sleep(0.5)
        
        # Aggregate predictions using majority voting on word level
        # Get all words from the original transcription with their positions
        words = target_file["transcription"].split()
        original_words = [w.lower().strip('.,!?') for w in words]
        
        # Count frequencies for each word position
        word_frequencies = []
        for i, word in enumerate(original_words):
            if word:  # Skip empty strings
                frequency = sum(1 for stressed_set in all_stressed_words_sets if word in stressed_set)
                word_frequencies.append(frequency)
            else:
                word_frequencies.append(0)
        
        # Use majority voting (at least 2 out of 3) to determine final stressed words
        # Create aggregated transcription with stress markers
        aggregated_transcription = []
        for i, word in enumerate(words):
            if i < len(word_frequencies) and word_frequencies[i] >= 3:
                aggregated_transcription.append(f"<*>{word}</*>")
            else:
                aggregated_transcription.append(word)
        aggregated_transcription = " ".join(aggregated_transcription)

        # # Use span-based accuracy calculation
        # accuracy, precision, recall, f1 = calculate_span_based_accuracy(
        #     aggregated_transcription, 
        #     target_file["intonation"],
        #     target_file["transcription"]
        # )
        
        # Also calculate traditional binary accuracy for comparison
        gt_binary_labels = target_file["stress_pattern"]["binary"]
        aggregated_binary = [1 if freq >= 3 else 0 for freq in word_frequencies]
        
        # Pad or trim to match ground truth length
        while len(aggregated_binary) < len(gt_binary_labels):
            aggregated_binary.append(0)
        aggregated_binary = aggregated_binary[:len(gt_binary_labels)]
        
        binary_accuracy = np.mean(np.array(aggregated_binary) == np.array(gt_binary_labels))
        binary_f1 = f1_score(gt_binary_labels, aggregated_binary) if len(aggregated_binary) == len(gt_binary_labels) else 0.0
        
        print(f"Aggregated stressed words for {target_file['audio_path']}: {aggregated_transcription}")
        print(f"  Individual runs: {all_transcriptions}")
        print(f"  Word frequencies: {word_frequencies}")
        print(f"  Binary-based - Accuracy: {binary_accuracy:.3f}, F1: {binary_f1:.3f}")
        
        # save the accuracy result to a folder
        folder_dir = "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstest/predict_results_aggregate_max_4"
        os.makedirs(folder_dir, exist_ok=True)
        result_path = os.path.join(folder_dir, os.path.basename(target_file["audio_path"]).replace(".wav", "_result.json"))
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {"binary_accuracy": binary_accuracy,
                 "binary_f1_score": binary_f1,
                 "source_file": target_file["audio_path"], 
                 "predicted_stressed_transcription": post_process_stress_markers(aggregated_transcription),
                 "individual_predictions": all_transcriptions,
                 "word_frequencies": word_frequencies,
                 "original_transcription": target_file["transcription"],
                 "original_stressed_transcription": target_file["intonation"]}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error processing {target_file}: {e}")

def cal_acc_all(target_files):
    # init
    genai.configure(api_key=API_KEY)

    # build prompt
    prompt_prefix = []
    prompt_prefix.append(sys_prompt)
    ICL_examples = []
    for ex in examples:
        example_file = upload_and_wait(ex["path"])
        ICL_examples.append(example_file)

        prompt_prefix.append("Input Audio:")
        prompt_prefix.append(example_file)
        prompt_prefix.append("Correct Output:")
        prompt_prefix.append(ex["ground_truth"])
    prompt_prefix.append("Now, analyze this audio file following the same format:")
    prompt_prefix.append("The ground truth transcription is given for reference only, do not rely on it to find stressed words.")

    model = genai.GenerativeModel(
        "gemini-3-pro-preview",
        generation_config={"response_mime_type": "application/json"}
    )

    total_acc = []
    for i, target_file in tqdm(enumerate(target_files)):
        target_file_obj = None

        # upload target audio
        target_file_obj = upload_and_wait(target_file["audio_path"])
        # prepare prompt
        current_prompt = prompt_prefix.copy()

        current_prompt.append(f'\nTranscription: "{target_file["transcription"]}"')
        current_prompt.append(target_file_obj)

        # inference
        response = model.generate_content(current_prompt)

        # post-process
        json_data = json.loads(response.text)
        json_data['source_file'] = target_file["audio_path"]
        print(f"Annotated stressed words for {target_file['audio_path']}: {json_data}")
        predicted_binary_labels = [0] * len(json_data['transcription'].split())
        for i, word in enumerate(json_data['transcription'].split()):
            if '<*>' in word and '</*>' in word:
                predicted_binary_labels[i] = 1

        gt_binary_labels = target_file["stress_pattern"]["binary"]
        accuracy = np.mean(np.array(predicted_binary_labels) == np.array(gt_binary_labels))
        total_acc.append(accuracy)

    print(f"Average accuracy: {np.mean(np.array(total_acc))}")

def create_new_stress_label_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        all_data = []
        for line in f:
            try:
                all_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}\n{e}")

    for i, data in enumerate(all_data):
        data['audio_path'] = f"/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstest/ground_truth/audio_{i}.wav"

    output_path = json_path.replace(".jsonl", "_with_audio_paths.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")

def acc_summary(predict_results_dir):
    all_acc = []
    all_f1 = []
    all_precision = []
    all_recall = []
    all_binary_acc = []
    all_binary_f1 = []
    
    for file_name in os.listdir(predict_results_dir):
        if file_name.endswith("_result.json"):
            with open(os.path.join(predict_results_dir, file_name), "r", encoding="utf-8") as f:
                data = json.load(f)
                if 'binary_accuracy' in data:
                    all_binary_acc.append(data['binary_accuracy'])
                if 'binary_f1_score' in data:
                    all_binary_f1.append(data['binary_f1_score'])

    print("\n=== Span-based Metrics (Multi-word aware) ===")
    print(f"Total average accuracy: {np.mean(np.array(all_acc)):.4f}")
    print(f"Total average precision: {np.mean(np.array(all_precision)):.4f}" if all_precision else "Precision: N/A")
    print(f"Total average recall: {np.mean(np.array(all_recall)):.4f}" if all_recall else "Recall: N/A")
    print(f"Total average f1 score: {np.mean(np.array(all_f1)):.4f}")
    
    if all_binary_acc:
        print("\n=== Binary Position-based Metrics (Original) ===")
        print(f"Total average binary accuracy: {np.mean(np.array(all_binary_acc)):.4f}")
        print(f"Total average binary f1 score: {np.mean(np.array(all_binary_f1)):.4f}")

if __name__ == "__main__":
    meta_data_path = "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstest/stresstest_with_audio_paths.jsonl"
    target_files = []
    with open(meta_data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if isinstance(data, dict):  # Ensure it's a dictionary
                    target_files.append(data)
                else:
                    print(f"Skipping invalid entry: {line.strip()}")
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line.strip()}\nError: {e}")
    # raise Exception("stop here for testing")
    # Flatten target_files if necessary
    flat_target_files = []
    for item in target_files:
        if isinstance(item, list):
            flat_target_files.extend(item)
        else:
            flat_target_files.append(item)

    # Debugging: Validate target_files before multiprocessing
    print("Validating target_files...")
    for i, entry in enumerate(flat_target_files):
        if not isinstance(entry, dict):
            print(f"Invalid entry at index {i}: {entry}")
            raise ValueError(f"Expected a dictionary at index {i}, but got {type(entry)}")

    # Check if target_files contains nested lists
    if isinstance(target_files[0], list):
        print("Flattening nested target_files...")
        target_files = [item for sublist in target_files for item in sublist]

    print("Validation complete. Proceeding with multiprocessing.")

    num_processes = multiprocessing.cpu_count()  # Use all available CPUs
    # num_processes = 16
    with multiprocessing.Pool(num_processes) as pool:
        list(tqdm(pool.imap(cal_acc_single, flat_target_files), total=len(flat_target_files)))

    # ========== get accuracy
    predict_results_dir = "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstest/predict_results_aggregate_max_4"
    acc_summary(predict_results_dir)