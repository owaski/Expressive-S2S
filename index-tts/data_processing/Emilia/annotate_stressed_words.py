import os
import sys
import google.generativeai as genai
import time
import json
import argparse
import glob
from tqdm import tqdm
import multiprocessing


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

# sys_prompt = """
#         You are a linguistic expert. Annotate the stress in the final audio file.
#         Study the following examples carefully to learn how to identify the stressed words only according to accoustic features. 
#         The stressed words tend to have high volume and high pitch. 
#         The transcription of the final audio file is given for reference only, do not rely on it to find stressed words.
#     """

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
def upload_and_wait(path, max_retries=3):
    """Upload a file with retry logic for SSL errors."""
    for attempt in range(max_retries):
        try:
            print(f"Uploading {path}... (attempt {attempt + 1}/{max_retries})")
            file_obj = genai.upload_file(path=path)
            
            # Wait for processing
            while file_obj.state.name == "PROCESSING":
                time.sleep(1)
                file_obj = genai.get_file(file_obj.name)
                
            if file_obj.state.name != "ACTIVE":
                raise ValueError(f"File {file_obj.name} failed to process.")
            return file_obj
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Upload failed with {type(e).__name__}: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Upload failed after {max_retries} attempts.")
                raise
    return None


def batch_annotate(audio_paths):
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

    for i, audio_path in tqdm(enumerate(audio_paths)):

        target_file_obj = None
        try:
            # upload target audio
            target_file_obj = upload_and_wait(audio_path)
            # prepare prompt
            current_prompt = prompt_prefix.copy()
            metadata_path = audio_path.replace("audio", "metadata").replace(".mp3", ".json")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                target_transcription = metadata["text"]
            current_prompt.append(f'\nTranscription: "{target_transcription}"')
            # print(current_prompt)
            current_prompt.append(target_file_obj)

            # TODO: inference, multiple tries, and aggregate the results
            response = model.generate_content(current_prompt)

            # post-process
            json_data = json.loads(response.text)
            file_name = os.path.basename(audio_path)
            json_data['source_file'] = file_name
            print(f"Annotated stressed words for {file_name}: {json_data}")
            metadata['stressed_words_transcription'] = json_data['transcription']

            # save the annotation to a new json file
            output_path = metadata_path.replace("diarized_metadata", "diarized_stress_metadata")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

            # raise Exception("stop here for testing")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
        finally:
            if target_file_obj:
                try:
                    genai.delete_file(target_file_obj.name)
                except Exception as e:
                    print(f"Failed to delete {target_file_obj.name}: {e}")

# Global variable to store uploaded example files (for multiprocessing)
_uploaded_examples = None

def init_worker(uploaded_examples_data):
    """Initialize worker process with pre-uploaded example files."""
    global _uploaded_examples
    _uploaded_examples = uploaded_examples_data
    genai.configure(api_key=API_KEY)

def annotate_single(audio_path, num_runs=5):
    global _uploaded_examples
    
    print(f"Processing {audio_path}...")

    # Use pre-uploaded examples instead of uploading again
    prompt_prefix = []
    prompt_prefix.append(sys_prompt)
    
    for i, ex in enumerate(examples):
        # Retrieve the file by name instead of uploading again
        example_file_name = _uploaded_examples[i]
        example_file = genai.get_file(example_file_name)
        
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

    try:
        # upload target audio
        target_file_obj = upload_and_wait(audio_path)
        current_prompt = prompt_prefix.copy()
        metadata_path = audio_path.replace("audio", "metadata").replace(".mp3", ".json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            target_transcription = metadata["text"]
        current_prompt.append(f'\nTranscription: "{target_transcription}"')
        current_prompt.append(target_file_obj)
        
        # inference
        all_stressed_words_sets = []
        all_transcriptions = []

        for run_idx in range(num_runs):
            print(f"  Run {run_idx + 1}/{num_runs} for {audio_path}")

            response = model.generate_content(current_prompt)

            # post-process
            json_data = json.loads(response.text)
            if len(json_data) > 0 and isinstance(json_data, list):
                json_data = json_data[0]
            all_transcriptions.append(json_data['transcription'])


            stressed_words = extract_stressed_words(json_data['transcription'])
            all_stressed_words_sets.append(stressed_words)

            if run_idx < num_runs - 1:
                time.sleep(0.5)
        words = target_transcription.split()
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
            if i < len(word_frequencies) and word_frequencies[i] >= (num_runs / 2):
                aggregated_transcription.append(f"<*>{word}</*>")
            else:
                aggregated_transcription.append(word)
        aggregated_transcription = " ".join(aggregated_transcription)

        print(f"  Aggregated Transcription for {audio_path}: {aggregated_transcription}")
        print(f"  Individual runs: {all_transcriptions}")
        print(f"  Word frequencies: {word_frequencies}")
        
        metadata['stressed_words_transcription'] = post_process_stress_markers(aggregated_transcription)
        metadata['all_transcriptions'] = all_transcriptions
        metadata['word_frequencies'] = word_frequencies
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"Error processing {audio_path}: {type(e).__name__}: {str(e)}")
    finally:
        # Clean up uploaded file
        if target_file_obj:
            try:
                genai.delete_file(target_file_obj.name)
            except Exception as e:
                print(f"Failed to delete {target_file_obj.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate stressed words in audio files.")
    parser.add_argument(
        "--base_dir",
        type=str,
        help="Base directory containing the audio files.",
        default="/data/user_data/willw2/data/EMILIA",
    )
    args = parser.parse_args()
    base_dir = args.base_dir
    shards = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
    target_files = []

    # # all shards
    # for shard in shards[:4]:
    #     files = glob.glob(f"{base_dir}/{shard}/diarized_audio/*.mp3")
    #     target_files.extend(files)

    # get the first shard
    shard = "EN-B000000"
    files = glob.glob(f"{base_dir}/{shard}/diarized_audio/*.mp3")
    target_files.extend(files)
    
    print(f"Total audio files to annotate: {len(target_files)}")
    print(f"the first audio file is {target_files[0]}")
    print(f"the last audio file is {target_files[-1]}")

    processed_audio_metadata_json = "/data/user_data/willw2/data/EMILIA/emilia_stressed_transcripted_audios_metadata.json"
    processed_audio_files = []
    with open(processed_audio_metadata_json, "r", encoding="utf-8") as f:
        processed_audio_metadata = json.load(f)
        for item in tqdm(processed_audio_metadata):
            processed_audio_files.append(item['audio_file'])
    
    target_files = [f for f in target_files if f not in processed_audio_files]
    print(f"Total audio files to annotate: {len(target_files)}")
    print(f"the first audio file is {target_files[0]}")
    print(f"the last audio file is {target_files[-1]}")
    # raise Exception("stop here for testing")

    print(f"Found {len(target_files)} audio files to annotate.")
    
    # Upload example files once before multiprocessing
    print("Uploading example files...")
    genai.configure(api_key=API_KEY)
    uploaded_example_names = []
    for ex in examples:
        try:
            example_file = upload_and_wait(ex["path"])
            uploaded_example_names.append(example_file.name)
            print(f"Uploaded {ex['path']} as {example_file.name}")
        except Exception as e:
            print(f"Failed to upload example {ex['path']}: {e}")
            raise
    
    print(f"All example files uploaded. Starting multiprocessing with {len(uploaded_example_names)} examples...")
    
    num_processes = multiprocessing.cpu_count()  # Use all available CPUs
    num_processes = min(64, num_processes)  # Reduced from 64 to 8 to avoid API rate limits
    with multiprocessing.Pool(num_processes, initializer=init_worker, initargs=(uploaded_example_names,)) as pool:
        list(tqdm(pool.imap(annotate_single, target_files), total=len(target_files)))

