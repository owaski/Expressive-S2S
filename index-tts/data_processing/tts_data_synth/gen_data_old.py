import os
# Unset TRANSFORMERS_CACHE to use HF_HOME instead (avoids cache conflicts)
if 'TRANSFORMERS_CACHE' in os.environ:
    del os.environ['TRANSFORMERS_CACHE']
import sys
sys.path.append('/data/user_data/willw2/CosyVoice/third_party/Matcha-TTS')
sys.path.append("/data/user_data/willw2/CosyVoice")
from cosyvoice.cli.cosyvoice import AutoModel
import json
import argparse
import random

# from openai import OpenAI
from google import genai
from pydantic import BaseModel
import torch
import torchaudio
import soundfile as sf
# from cosyvoice.cli.cosyvoice import AutoModel

# from qwen_tts import Qwen3TTSModel
from tqdm import tqdm
import re
import random
import torch.multiprocessing as mp
import glob

from prompt import emo_speaker_prompt, SENTENCE_TYPES, Emo1Stress2, Emo2Stress1, Emo2Stress2, data_gen_config, sentence_domain_topic_data

try:
    import flash_attn  # noqa: F401
    _HAS_FLASH_ATTN = True
except Exception:
    _HAS_FLASH_ATTN = False

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


# OPENAI_KEY = "YOUR KEY HERE"
OPENAI_KEY = None
GEMINI_KEY="AIzaSyDhidZLY2k1SUQlYr5JJBH42hi4W0tmgYs"


class TextIntention(BaseModel):
    original_text: str
    stressed_text_1: str
    intention_1: str
    emotion_1: str
    stressed_text_2: str
    intention_2: str
    emotion_2: str

translation_prompt_template = """
    You are a professional translator. Translate the following English text to Simplified Chinese. CRITICAL: You must preserve all Markdown formatting (like asterisks for bolding) and apply it to the corresponding Chinese words. Translate the following 5 English text to Chinese: 
    
    original_text: {}
    stressed_text_1: {}
    intention_1: {}
    stressed_text_2: {}
    intention_2: {}
"""


instruct_prompts = {
    "happy": "Please speak the sentence very happily",
    "sad": "Please speak the sentence very sadly",
    "default": "please speak the sentence in a default way."
}

def gen_text_gemini(num, domain_topic, config_type, emo1, emo2, text_dir):
    gemini_client = genai.Client(api_key=GEMINI_KEY)

    for i in range(num):
        domain = random.choice(list(domain_topic.keys()))
        topic = random.choice(domain_topic[domain])
        sentence_type = random.choice(SENTENCE_TYPES)
        prompt = data_gen_config[config_type]['prompt'].format(emo_1=emo1, emo_2=emo2, domain=domain, topic=topic)

        # raise NotImplementedError("This function is a work in progress and contains placeholder code. Please refer to the latest version for the implemented functionality.")
        response = gemini_client.models.generate_content(
            model="gemini-3-pro-preview", # Changed from gemini-3-flash-preview as per standard model names or keep user's if valid
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": data_gen_config[config_type]["return_class"].model_json_schema(),
            },
        )

        result = data_gen_config[config_type]["return_class"].model_validate_json(response.text)
        # print(result)
        if config_type == "emo1stress2":
            data = {
                "domain": domain,
                "topic": topic,
                "sentence_type": sentence_type,
                "original_text": result.original_text,
                'emotion_1': result.emotion_1,
                "stressed_text_1": result.stressed_text_1,
                "intention_e1_s1": result.intention_e1_s1,
                "stressed_text_2": result.stressed_text_2,
                "intention_e1_s2": result.intention_e1_s2,
            }
        elif config_type == "emo2stress1":
            data = {
                "domain": domain,
                "topic": topic,
                "sentence_type": sentence_type,
                "original_text": result.original_text,
                'emotion_1': result.emotion_1,
                'emotion_2': result.emotion_2,
                "stressed_text_1": result.stressed_text_1,
                "intention_e1_s1": result.intention_e1_s1,
                "intention_e2_s1": result.intention_e2_s1,
            }
        elif config_type == "emo2stress2":
            data = {
                "domain": domain,
                "topic": topic,
                "sentence_type": sentence_type,
                "original_text": result.original_text,
                'emotion_1': result.emotion_1,
                'emotion_2': result.emotion_2,
                "stressed_text_1": result.stressed_text_1,
                "stressed_text_2": result.stressed_text_2,
                "intention_e1_s1": result.intention_e1_s1,
                "intention_e1_s2": result.intention_e1_s2,
                "intention_e2_s1": result.intention_e2_s1,
                "intention_e2_s2": result.intention_e2_s2
            }
        else:
            raise ValueError(f"Invalid config_type: {config_type}")

        # save to a json file
        with open(os.path.join(text_dir, f'text_{i}.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

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
    cosyvoice = AutoModel(model_dir=model_dir)
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
                intention = TTS_text["intention"].replace("The speaker is ", "")
                for model_output in cosyvoice.inference_instruct2(
                    tts_text=tts_text,
                    instruct_text=f"{TTS_text['instruct_prompt']}<|endofprompt|>",
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

    num_workers = 2
    print(f"Detected {num_workers} GPUs, using {num_workers} workers")
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
    text_files = glob.glob(os.path.join(text_jsons, '*.json'))
    for text_file in tqdm(text_files):
        # try:
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
        else:
            raise ValueError(f"Invalid config_type: {config_type}")
        # except Exception as e:
        #     print(f"Error writing tqdm output for {text_file}: {e}")

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
        # Single worker mode
        cosyvoice_inference_worker(0, 1, TTS_texts, output_dir, n_pass, model_dir)
    
    print(f"Processing complete! Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using OpenAI API")
    parser.add_argument(
        "--num", 
        type=int, 
        default=10, 
        help="Number of texts to generate")

    parser.add_argument(
        "--audio_dir", 
        type=str, 
        default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/audio_emo2stress1_direct", 
        help="Path to save generated audio")

    parser.add_argument(
        "--text_dir", 
        type=str, 
        default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/text_emo2stress1_direct",
        help="Directory containing text files for speech generation")
    
    
    parser.add_argument(
        "--n_pass", 
        type=int, 
        default=10,
        help="Number of passes for speech generation")
    
    parser.add_argument(
        "--config_type",
        type=str,
        default="emo2stress1",
        choices=["emo1stress2", "emo2stress1", "emo2stress2"],
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

    args = parser.parse_args()
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(args.text_dir, exist_ok=True)

    # gen_text_chatgpt(args.num, sentence_domain_topic_data, prompt_template, args.text_dir)
    gen_text_gemini(args.num, sentence_domain_topic_data, args.config_type, args.emo1, args.emo2, args.text_dir)
    gen_speech_cosyvoice3(
        text_jsons=args.text_dir,
        config_type=args.config_type,
        output_dir=args.audio_dir,
        n_pass=args.n_pass
    )