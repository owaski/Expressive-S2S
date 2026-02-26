import os
# Unset TRANSFORMERS_CACHE to use HF_HOME instead (avoids cache conflicts)
if 'TRANSFORMERS_CACHE' in os.environ:
    del os.environ['TRANSFORMERS_CACHE']
import sys
import re
import json
import argparse
import random
import glob
import string
from google import genai
from google.genai import types
from tqdm import tqdm
import torch.multiprocessing as mp
from tqdm import tqdm
from prompt import emo_speaker_prompt, data_gen_config, GEMINI_KEY, distribution_stress_word_selection_prompt
import numpy as np
import torch
import torchaudio
import librosa
sys.path.append('/data/user_data/willw2/CosyVoice/third_party/Matcha-TTS')
sys.path.append("/data/user_data/willw2/CosyVoice")
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
from cosyvoice.cli.cosyvoice import AutoModel as CosyVoiceAutoModel

def extract_stressed_word(text):
    return text.split("*")[1]

def asterisk_to_strong(text):
    """Convert text with *word* to <strong>word</strong>"""
    return re.sub(r'\*([^*]+)\*', r'<strong>\1</strong>', text)

def extract_stressed_word_zh(text):
    """
    Extracts all substrings enclosed in asterisks.
    Example: "这是*一个*测试*句子*" -> ['一个', '句子']
    """
    pattern = r'\*(.*?)\*'
    
    return re.findall(pattern, text)

def format_stressed_text(json_data, config_type):
    """
    Takes a parsed JSON dictionary and returns an HTML-formatted string
    with the underlying stress pattern and alternative stress target wrapped
    in <strong> tags.
    """
    # 1. Split the text into a list of words
    words = json_data['tts_text'].split()
    
    # 2. Gather the indices of the underlying stress pattern
    stress_indices = set()
    for item in json_data.get('underlying_stress_pattern', []):
        stress_indices.add(item['word_index'])
        
    # 3. Add the index of the alternative stress target (if it exists)
    if config_type == "base_pattern_extra":
        alt_target = json_data.get('alternative_stress_target')
        if alt_target and 'word_index' in alt_target:
            stress_indices.add(alt_target['word_index'])
        
    # 4. Wrap the target words in <strong> tags
    formatted_words = []
    for i, word in enumerate(words):
        if i in stress_indices:
            formatted_words.append(f"<strong>{word}</strong>")
        else:
            formatted_words.append(word)
            
    # 5. Join back into a single string
    return " ".join(formatted_words)

def collect_stressed_pattern_files(distribution_dir):
    json_files = glob.glob(os.path.join(distribution_dir, '*/*.json'))
    return json_files


def collect_stress_patterns_in_place(file_paths):
    """
    Processes a list of JSON file paths to calculate the adaptive stress 
    threshold, extracts the underlying pattern, and overwrites the 
    original files with the new keys.
    """
    successful_updates = 0
    
    for file_path in tqdm(file_paths):
        try:
            # 1. Read the existing JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 2. Clean punctuation to map text to the distribution array
            # Note: adjust the replacement logic if your TTS handles hyphens/apostrophes differently
            clean_text = data['tts_text'].translate(str.maketrans('', '', string.punctuation))
            words = clean_text.split()
            dist = data['stressed_word_distribution']
            
            # 3. Calculate the adaptive threshold
            max_stress = max(dist) if dist else 0
            threshold = max_stress * 0.5
            
            # 4. Extract the underlying pattern
            underlying_pattern = []
            for i, (word, score) in enumerate(zip(words, dist)):
                if score >= threshold:
                    underlying_pattern.append({
                        "word_index": i,
                        "word": word,
                        "stress_score": score
                    })
            
            # 5. Append the new information to the dictionary
            data['adaptive_threshold'] = threshold
            data['underlying_stress_pattern'] = underlying_pattern
            
            # 6. Write the modified dictionary back to the exact same file path
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            successful_updates += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    print(f"Successfully updated {successful_updates} out of {len(file_paths)} files.")
    return successful_updates

def generate_alternative_stress(json_paths, api_key):
    """
    Reads JSON files, prompts Gemini to find an alternative word to stress,
    and updates the JSON files in-place with the AI's selection.
    """
    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)
    
    # Define the exact JSON structure we want Gemini to return
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "selected_word": types.Schema(
                type=types.Type.STRING,
                description="The alternative word chosen to be stressed."
            ),
            "word_index": types.Schema(
                type=types.Type.INTEGER,
                description="The 0-based index of the selected word in the sentence."
            ),
            "reasoning": types.Schema(
                type=types.Type.STRING,
                description="A brief 1-sentence explanation of why stressing this word fits the emotion."
            )
        },
        required=["selected_word", "word_index", "reasoning"]
    )

    successful_updates = 0

    for path in tqdm(json_paths):
        try:
            # 1. Read the current data
            with open(path, 'r') as f:
                data = json.load(f)
                
            text = data.get('tts_text', '')
            emotion = data.get('emotion', 'Neutral')
            
            # Safely extract the existing underlying pattern we generated earlier
            underlying_pattern = data.get('underlying_stress_pattern', [])
            stressed_words = [item['word'] for item in underlying_pattern]
            
            # 2. Build the prompt dynamically
            prompt = f"""
                You are an expert in English phonetics, prosody, and emotional speech synthesis. 

                I am generating text-to-speech (TTS) audio with a "{emotion}" emotion. 
                Here is the sentence: "{text}"

                Currently, the TTS engine naturally places primary or secondary stress on these words (the underlying pattern): {stressed_words}

                Your task is to select exactly ONE new word from the sentence to receive deliberate phonetic stress. This new stress must sound natural for a human speaker conveying a "{emotion}" emotion.

                RULES:
                1. You MUST NOT select a word from the underlying pattern list.
                2. You MUST NOT select function words (e.g., articles like "the", "a"; prepositions like "to", "from", "of"; auxiliary verbs like "was", "is"; or basic pronouns like "I", "you", "we"), unless shifting the stress to a pronoun completely changes the pragmatic meaning in a way that fits the emotion.
                3. You MUST select a content word (e.g., a noun, main verb, adjective, or adverb) that adds nuance, contrast, or intensity to the emotion.
            """
            
            # 3. Call the Gemini API 
            # Note: Update 'gemini-3.0-flash-preview' if you are using a different model version identifier (e.g. gemini-2.5-flash)
            response = client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.4 # Lower temperature keeps the linguistic reasoning more grounded
                )
            )
            
            # 4. Parse the response and update the dictionary
            ai_selection = json.loads(response.text)
            data['alternative_stress_target'] = ai_selection
            
            # 5. Save it back to the file
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
                
            print(f"Success [{path}]: Gemini selected -> '{ai_selection['selected_word']}'")
            successful_updates += 1
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
    print(f"\nFinished! Successfully updated {successful_updates} out of {len(json_paths)} files.")

def cosyvoice_inference_worker(rank, world_size, TTS_texts, output_dir, n_pass, model_dir):
    """
    Worker function that initializes the model once and processes all assigned files.
    Each worker handles a subset of target files assigned using round-robin distribution.
    """
    # CRITICAL: Set the default CUDA device for this process at the very beginning
    # This prevents model components from leaking to cuda:0
    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    
    print(f"Worker {rank} strictly locked to device {device}")
    
    # Load model once per worker
    print(f"Worker {rank}: Loading CosyVoice model...")
    cosyvoice = CosyVoiceAutoModel(model_dir=model_dir, load_trt=True, load_vllm=True, fp16=False)
    print(f"Worker {rank}: Model loaded. Starting processing...")
    
    # Process assigned subset of files (round-robin distribution)
    # Store global indices to avoid audio_id collisions between workers
    my_indices = list(range(rank, len(TTS_texts), world_size))
    my_TTS_texts = [(idx, TTS_texts[idx]) for idx in my_indices]
    
    print(f"Worker {rank}: Processing {len(my_TTS_texts)} files...")
    
    for global_idx, TTS_text in tqdm(my_TTS_texts, desc=f"Worker {rank}", total=len(my_TTS_texts)):
        try:
            audio_id = f"audio_{global_idx}"
            audio_id_dir = os.path.join(output_dir, audio_id)
            os.makedirs(audio_id_dir, exist_ok=True)

            transcription = TTS_text["stressed_text"]
            tts_text = asterisk_to_strong(transcription)

            print(f"Worker {rank}: Generating audio for {audio_id} with transcription: {tts_text}")
            for i in range(n_pass):
                metadata = TTS_text.copy()
                chunks = []
                # intention = TTS_text["intention"].replace("The speaker is ", "")
                for model_output in cosyvoice.inference_instruct2(
                    tts_text=tts_text,
                    instruct_text= f"You are a helpful assistant. Please speak in a {TTS_text['emotion']} mood.<|endofprompt|>", # f"{TTS_text['instruct_prompt']}<|endofprompt|>",
                    prompt_wav=TTS_text["speaker_prompt"], stream=False
                ):
                    chunks.append(model_output["tts_speech"])
                
                if not chunks:
                    print(f"Worker {rank}: No audio generated for {audio_id}, skipping.")
                    continue

                tts_speech = torch.cat(chunks, dim=1)
                audio_path = os.path.join(audio_id_dir, f'{audio_id}_{i}.wav')
                torchaudio.save(
                    audio_path, 
                    tts_speech, 
                    cosyvoice.sample_rate
                )
                metadata['audio_path'] = audio_path
                with open(os.path.join(audio_id_dir, f'{audio_id}_{i}.json'), "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Worker {rank}: Error processing {TTS_text['audio_path']}: {e}")
    
    print(f"Worker {rank}: Completed all files!")

def gen_speech_cosyvoice3(text_jsons, config_type, output_dir, n_pass):
    """CosyVoice3 Usage with multiprocessing using mp.spawn pattern"""

    num_workers = 1
    print(f"Using {num_workers} worker (vLLM manages GPU internally)")
    model_dir = '/data/user_data/willw2/CosyVoice/pre_trained_models/Fun-CosyVoice3-0.5B'
    
    """ Collect metadata for all TTS texts
        For example 
        {
            "domain": "Education",
            "topic": "Online Learning",
            "sentence_type": "Statement",
            "original_text": "Tell the board the quantum experiment is done"
            "stressed_text": "Tell the *board* the quantum experiment is done",
            "intention": "The speaker is eagerly instructing a colleague to inform leadership that the project has been successfully completed.",
            "emotion": "happy",
            "instruct_prompt": "please speak the sentence very happily.",
            "speaker_prompt": "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/eval/expresso/audio_48khz/read/ex03/happy/base/ex03_happy_00165.wav"
        }
    """
    TTS_texts = []
    text_files = []
    if config_type in ["emo1stress2", "emo2stress1", "emo2stress2"]:
        print(f"Loading TTS texts for config type: {config_type} from {text_jsons}")
        text_files = glob.glob(os.path.join(text_jsons, '*.json'))
    elif config_type == "base_pattern_extra" or config_type == "base_pattern_base":
        print(f"Loading TTS texts for {config_type} config from {text_jsons}")
        text_files = collect_stressed_pattern_files(text_jsons)
    for text_file in tqdm(text_files):
        try:
            tqdm.write(f"Processing text file for TTS metadata: {text_file}")
            with open(text_file, 'r', encoding='utf-8') as f:
                d = json.load(f)
            if config_type == "emo1stress2":
                for s_i in [1, 2]:
                    cur_tts_text = {}
                    cur_tts_text["domain"] = d['domain']
                    cur_tts_text["topic"] = d['topic']
                    cur_tts_text["sentence_type"] = d['sentence_type']
                    cur_tts_text["original_text"] = d['original_text']
                    cur_tts_text["stressed_text"] = d[f'stressed_text_{s_i}']
                    intention_key = f'intention_e1_s{s_i}'
                    cur_tts_text["intention"] = d[intention_key]
                    emotion = d['emotion_1'].lower()
                    cur_tts_text["emotion"] = emotion
                    cur_tts_text["instruct_prompt"] = data_gen_config[config_type]['tts_instruct_prompt'].format(emo=emotion, intention=d[intention_key])
                    cur_tts_text["speaker_prompt"] = random.choice(emo_speaker_prompt[emotion])["audio_path"]
                    TTS_texts.append(cur_tts_text)
            elif config_type == "emo2stress1":
                for e_i in [1, 2]:
                    cur_tts_text = {}
                    cur_tts_text["domain"] = d['domain']
                    cur_tts_text["topic"] = d['topic']
                    cur_tts_text["sentence_type"] = d['sentence_type']
                    cur_tts_text["original_text"] = d['original_text']
                    cur_tts_text["stressed_text"] = d['stressed_text_1']
                    intention_key = f'intention_e{e_i}_s1'
                    cur_tts_text["intention"] = d[intention_key]
                    emotion = d[f'emotion_{e_i}'].lower()
                    cur_tts_text["emotion"] = emotion
                    cur_tts_text["instruct_prompt"] = data_gen_config[config_type]['tts_instruct_prompt'].format(emo=emotion, intention=d[intention_key])
                    cur_tts_text["speaker_prompt"] = random.choice(emo_speaker_prompt[emotion])["audio_path"]
                    TTS_texts.append(cur_tts_text)
            elif config_type == "emo2stress2":
                for e_i in [1, 2]:
                    for s_i in [1, 2]:
                        cur_tts_text = {}
                        cur_tts_text["domain"] = d['domain']
                        cur_tts_text["topic"] = d['topic']
                        cur_tts_text["sentence_type"] = d['sentence_type']
                        cur_tts_text["original_text"] = d['original_text']
                        cur_tts_text["stressed_text"] = d[f'stressed_text_{s_i}']
                        intention_key = f'intention_e{e_i}_s{s_i}'
                        cur_tts_text["intention"] = d[intention_key]
                        emotion = d[f'emotion_{e_i}'].lower()
                        cur_tts_text["emotion"] = emotion
                        cur_tts_text["instruct_prompt"] = data_gen_config[config_type]['tts_instruct_prompt'].format(emo=emotion, intention=d[intention_key])
                        cur_tts_text["speaker_prompt"] = random.choice(emo_speaker_prompt[emotion])["audio_path"]
                        TTS_texts.append(cur_tts_text)
            elif config_type == "base_pattern_extra" or config_type == "base_pattern_base":
                cur_tts_text = {}
                # cur_tts_text['speaker_prompt'] = d['speaker_prompt']
                cur_tts_text['speaker_prompt'] = random.choice(emo_speaker_prompt[d['emotion'].lower()])["audio_path"]
                cur_tts_text['original_text'] = d['tts_text']
                cur_tts_text['stressed_text'] = format_stressed_text(d, config_type)
                print(f"Formatted stressed text for {text_file}: {cur_tts_text['stressed_text']}")
                cur_tts_text['intention'] = d['alternative_stress_target']['reasoning']
                cur_tts_text['emotion'] = d['emotion']
                cur_tts_text['instruct_prompt'] = distribution_stress_word_selection_prompt.format(emo=d['emotion'], intention=d['alternative_stress_target']['reasoning'])
                TTS_texts.append(cur_tts_text)
            else:
                raise ValueError(f"Invalid config_type: {config_type}")
        except Exception as e:
            print(f"Error writing tqdm output for {text_file}: {e}")

    print(f"Loaded {len(TTS_texts)} TTS texts. Starting multiprocessing...")

    # Use mp.spawn for proper CUDA multiprocessing
    if num_workers > 1:
        mp.spawn(
            cosyvoice_inference_worker,
            args=(num_workers, TTS_texts, output_dir, n_pass, model_dir),
            nprocs=num_workers,
            join=True
        )
    else:
        cosyvoice_inference_worker(0, 1, TTS_texts, output_dir, n_pass, model_dir)
    
    print(f"Processing complete! Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using OpenAI API")
    # parser.add_argument(
    #     "--num_data", 
    #     type=int, 
    #     default=10, 
    #     help="Number of texts to generate")

    parser.add_argument(
        "--audio_dir", 
        type=str, 
        default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/audio_distribution_base_base_word_emo_spk_random", 
        help="Path to save generated audio")

    parser.add_argument(
        "--text_dir", 
        type=str, 
        # default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/text_gemini_emo_aware",
        default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/text_gemini_emo_aware_stress_distribution",
        help="Directory containing text files for speech generation")

    parser.add_argument(
        "--libritts_metadata_json",
        type=str,
        default="/data/user_data/willw2/data/LibriTTS_R/libritts_r_testset_metadata.json",
        help="Path to the LibriTTS metadata JSON file")

    # parser.add_argument(
    #     "--sample_size_for_distribution",
    #     type=int,
    #     default=100,
    #     help="Number of samples to analyze from LibriTTS for stressed word distribution")

    parser.add_argument(
        "--stress_distribution_output_dir",
        type=str,
        default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/text_gemini_emo_aware_stress_distribution",
        help="Directory to save the stressed word distribution results")

    parser.add_argument(
        "--n_pass", 
        type=int, 
        default=10,
        help="Number of passes for speech generation")
    
    parser.add_argument(
        "--config_type",
        type=str,
        default="base_pattern_base",
        choices=["emo1stress2", "emo2stress1", "emo2stress2", "base_pattern_extra", "base_pattern_base"],
        help="Configuration type for text generation (default: emo2stress2)"
    )

    parser.add_argument(
        "--emo1",
        type=str,        
        default="happy",
        choices=["happy", "sad"],
        help="Emotion 1 for text generation (default: happy)"
    )

    parser.add_argument(
        "--emo2",
        type=str,        
        default="sad",
        choices=["happy", "sad"],
        help="Emotion 2 for text generation (default: sad)"
    )

    parser.add_argument(
        '--stress_threshold',
        type=float,
        default=0.7,
        help="The probability threshold for selecting stressed words from distribution"
    )

    args = parser.parse_args()
    os.makedirs(args.audio_dir, exist_ok=True)
    # os.makedirs(args.text_dir, exist_ok=True)
    os.makedirs(args.stress_distribution_output_dir, exist_ok=True)
    
    # gen 1000 audios with libritts speaker prompts, with different emotions, and calculate the distribution of detected stressed words using whistress.
    # get_stressed_word_distribution(args.libritts_metadata_json, args.sample_size_for_distribution, args.text_dir, args.stress_distribution_output_dir)
    print(f"number of collected files for TTS: {len(collect_stressed_pattern_files(args.stress_distribution_output_dir))}")
    distribution_files = collect_stressed_pattern_files(args.stress_distribution_output_dir)
    # collect_stress_patterns_in_place(distribution_files)
    # generate_alternative_stress(distribution_files, GEMINI_KEY)

    gen_speech_cosyvoice3(
        text_jsons=args.text_dir,
        config_type=args.config_type,
        output_dir=args.audio_dir,
        n_pass=args.n_pass
    )