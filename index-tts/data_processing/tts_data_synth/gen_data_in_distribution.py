import os
# Unset TRANSFORMERS_CACHE to use HF_HOME instead (avoids cache conflicts)
if 'TRANSFORMERS_CACHE' in os.environ:
    del os.environ['TRANSFORMERS_CACHE']
import sys
sys.path.append('/data/user_data/willw2/CosyVoice/third_party/Matcha-TTS')
sys.path.append("/data/user_data/willw2/CosyVoice")
sys.path.append("/data/user_data/willw2/course_project_repo/Expressive-S2S/index-tts/WhiStress")
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel as CosyVoiceAutoModel
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm
# from whistress import WhiStressInferenceClient
import json
import argparse
import random
import numpy as np

# from openai import OpenAI
from google import genai
from google.genai import types
from pydantic import BaseModel
import torch
import torchaudio
import soundfile as sf
import librosa
from funasr import AutoModel as FunASRAutoModel


from tqdm import tqdm
import re
import random
import torch.multiprocessing as mp
import glob

from prompt import (
    emo_speaker_prompt, SENTENCE_TYPES,
    Emo1Stress2, Emo2Stress1, Emo2Stress2,
    data_gen_config, sentence_domain_topic_data,
    OPENAI_KEY, GEMINI_KEY,
    tts_translation_prompt, stress_prediction_schema,
    translation_schema, gemini_stress_prediction_prompt,
    lang_configs
)

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

def get_word_account(lang_code, text):
    if lang_code == "en":
        text = re.findall(r"[\w']+", text)
        return text, len(text)
    elif lang_code == "zh":
        text = re.sub(r'[^\w\u4e00-\u9fff]+', '', text)
        return text, len(text)

def get_stressed_word_distribution_target_language(
        libritts_metadata_json, 
        wenetspeech_metadata_json,
        sample_size_for_distribution, 
        n_pass_for_stress_prediction, 
        tts_text_dir, 
        stress_distribution_output_dir, 
        target_language_code
    ):
    client = genai.Client(api_key=GEMINI_KEY)
    model_dir = '/data/user_data/willw2/CosyVoice/pre_trained_models/Fun-CosyVoice3-0.5B'
    if target_language_code == "zh":
        with open(wenetspeech_metadata_json, 'r') as f:
            speaker_prompts = json.load(f)
    elif target_language_code == "en":
        with open(libritts_metadata_json, 'r') as f:
            speaker_prompts = json.load(f)
    else:
        raise ValueError(f"Unsupported target language code: {target_language_code}. Currently only 'zh' and 'en' are supported.")
    speaker_prompts = [m for m in speaker_prompts if m.get('duration_sec', 0) > 3 and m.get('duration_sec', 0) < 10]

    # init TTS model, vLLM
    cosyvoice = CosyVoiceAutoModel(model_dir=model_dir, load_trt=True, load_vllm=True, fp16=False)
    # whistress_client = WhiStressInferenceClient()
    # init emotion recognition model for emotion score
    emo2vec_model = FunASRAutoModel(model="iic/emotion2vec_plus_large", hub="hf")

    # create two directories for two emotions: happy and sad, happy should be emotion_1 and sad should be emotion_2 
    for emotion in ["happy", "sad"]:
        emotion_dir = os.path.join(stress_distribution_output_dir, f'{emotion}_{target_language_code}')
        os.makedirs(emotion_dir, exist_ok=True)

    # collect TTS texts for stress words distribution estimation    
    tts_jsons = glob.glob(os.path.join(tts_text_dir, "*.json"))
    all_tts_data = []
    for tts_json in tts_jsons:
        with open(tts_json, 'r', encoding='utf-8') as f:
            tts_text = json.load(f)
        all_tts_data.append(tts_text)
    
    for d_id, tts_data in enumerate(all_tts_data):
        # random sample a speaker prompt, fix speaker prompt for each distribution estimation
        speaker_info = random.choice(speaker_prompts)
        speaker_prompt = speaker_info['audio_path']
        speaker_prompt_text = speaker_info['transcription']

        for emotion in ["happy", "sad"]:
            stressed_distribution_subdir = os.path.join(stress_distribution_output_dir, f'{emotion.lower()}_{target_language_code}', f"d_{d_id}")
            os.makedirs(stressed_distribution_subdir, exist_ok=True)
            stressed_word_distribution = []
            try_count = 0
            if target_language_code == 'zh':
                emo_prompt = "开心" if emotion == "happy" else "难过"
                tts_instruct_prompt = f"You are a helpful assistant. 请非常{emo_prompt}地说一句话。<|endofprompt|>"
            elif target_language_code == 'en':
                emo_prompt = emotion
                tts_instruct_prompt = f"You are a helpful assistant. Please say the sentence in a very {emo_prompt} mood.<|endofprompt|>"
            else:
                raise ValueError(f"Unsupported target language code: {target_language_code}. Currently only 'zh' and 'en' are supported.")

            tts_text = tts_data[f"tts_text_{target_language_code}"] # "Tell the board the quantum experiment is done"
            normalized_text, word_count = get_word_account(target_language_code, tts_text)
            prompt_config = lang_configs[target_language_code]
            stress_pred_prompt = gemini_stress_prediction_prompt.format(
                language_name=prompt_config['name'],
                language_code=target_language_code,
                language_specific_note=prompt_config['note'],
                transcription=normalized_text,
                word_count=word_count
            )
            # make sure collect sample_size number of samples that have emotion score > 0.9, 
            # # if the emotion score is not high enough, keep generating until we get enough samples or try_count reaches 10 to avoid infinite loop
            while len(stressed_word_distribution) < sample_size_for_distribution:
                sample_id = len(stressed_word_distribution)
                print(f"for the current sample{sample_id}, the tts_text is {tts_text}, the speaker prompt is {speaker_prompt}, the stress prediction prompt is {stress_pred_prompt}")
                # TTS
                chunks = []
                for model_output in cosyvoice.inference_instruct2(
                    tts_text=tts_text,
                    instruct_text=tts_instruct_prompt,
                    prompt_wav=speaker_prompt, stream=False
                ):
                    chunks.append(model_output["tts_speech"])
                tts_speech = torch.cat(chunks, dim=1)
                audio_path = os.path.join(stressed_distribution_subdir, f'{emotion.lower()}_sample_{d_id}_{sample_id}.wav')
                torchaudio.save(audio_path, tts_speech, cosyvoice.sample_rate)
                # emotion score
                rec_result = emo2vec_model.generate(audio_path, output_dir=None, granularity="utterance", extract_embedding=True)
                if emotion.lower() == "happy":
                    emo_key = '开心/happy' # the key of emotion scores in FunASR's output dictionary 
                elif emotion.lower() == "sad":
                    emo_key = '难过/sad' # the key of emotion scores in FunASR's output dictionary
                emotion_idx = rec_result[0]['labels'].index(emo_key) # Get emotion score: find the index of the emotion in labels list
                emotion_score = rec_result[0]['scores'][emotion_idx]
                print(f"{emo_key} Emotion score for {audio_path}: {emotion_score}")
                if emotion_score <= 0.9:
                    print(f"Emotion score {emotion_score} is not high enough for sample {sample_id}, trying again...")
                    try_count += 1
                    if try_count >= 10:
                        print(f"Failed to get enough stressed word samples for emotion {emotion} after 10 attempts, moving on to next sentence.")
                        break
                    continue

                # stress prediction
                uploaded_audio = client.files.upload(file=audio_path)
                n_pass_predictions = []
                for i in range(n_pass_for_stress_prediction):
                    response = client.models.generate_content(
                                model='gemini-3-flash-preview', 
                                contents=[uploaded_audio, stress_pred_prompt], # Pass BOTH the audio file object and the text prompt
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    response_schema=stress_prediction_schema,
                                    temperature=1.0
                                ),
                            )
                    stress_prediction_result = json.loads(response.text)
                    pred_pattern = stress_prediction_result["stress_pattern"]
                    n_pass_predictions.append(pred_pattern)

                pred_stress_pattern = n_pass_predictions
                if emotion_score > 0.9:
                    stressed_word_distribution.append(pred_stress_pattern)
                    try_count = 0
                    sample_info = {
                        "speaker_prompt": speaker_prompt,
                        "audio_path": audio_path,
                        "transcription": tts_text,
                        "pred_stress_pattern": pred_stress_pattern,
                        "emotion_score": emotion_score,
                    }
                    with open(os.path.join(stressed_distribution_subdir, f'{emotion.lower()}_sample_{d_id}_{sample_id}_info.json'), 'w') as f:
                        json.dump(sample_info, f, indent=4, ensure_ascii=False)
                else:
                    try_count += 1
                    if try_count >= 10:
                        print(f"Failed to get enough stressed word samples for emotion {emotion} after 10 attempts, moving on to next sentence.")
                        break

        # calculate the distribution of stressed words
        if len(stressed_word_distribution) < sample_size_for_distribution:
            print(f"Skipping distribution calculation for emotion {emotion}, idx {d_id}: only {len(stressed_word_distribution)}/{sample_size_for_distribution} samples collected.")
            continue
        stressed_word_ndarray = np.array(stressed_word_distribution)
        stressed_word_freq = np.sum(stressed_word_ndarray, axis=0)
        distribution_info = {
            "emotion": emotion,
            "sample_size": sample_size_for_distribution,
            "tts_text": tts_text,
            "speaker_prompt": speaker_prompt,
            "stressed_word_freq": stressed_word_freq.tolist(),
            "stressed_word_distribution": (stressed_word_freq / sample_size_for_distribution).tolist()
        }
        with open(os.path.join(stress_distribution_output_dir, f'{emotion.lower()}_{target_language_code}', f'{d_id}_stressed_word_distribution.json'), 'w') as f:
            json.dump(distribution_info, f, indent=4)

def worker(gpu_id, text_subset, args, target_language_code):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    # Initialize models INSIDE the worker so they load onto the specific GPU
    model_dir = '/data/user_data/willw2/CosyVoice/pre_trained_models/Fun-CosyVoice3-0.5B'
    cosyvoice = CosyVoiceAutoModel(model_dir=model_dir, load_trt=True, load_vllm=True, fp16=False)
    emo2vec_model = FunASRAutoModel(model="iic/emotion2vec_plus_large", hub="hf")
    
    # Setup Gemini Client
    client = genai.Client(api_key=GEMINI_KEY)

    # Load speaker prompts (logic from your original function)
    if target_language_code == "zh":
        with open(args.wenetspeech_metadata_json, 'r') as f:
            speaker_prompts = json.load(f)
    else:
        with open(args.libritts_metadata_json, 'r') as f:
            speaker_prompts = json.load(f)
    speaker_prompts = [m for m in speaker_prompts if 3 < m.get('duration_sec', 0) < 10]

    # Process the subset assigned to this GPU
    for d_id, tts_data in text_subset:
        for emotion in ["happy", "sad"]:
            stressed_distribution_subdir = os.path.join(args.stress_distribution_output_dir, f'{emotion.lower()}_{target_language_code}', f"d_{d_id}")
            os.makedirs(stressed_distribution_subdir, exist_ok=True)
            
            stressed_word_distribution = []
            try_count = 0
            
            # 1. Setup Prompts
            if target_language_code == 'zh':
                emo_prompt = "开心" if emotion == "happy" else "难过"
                tts_instruct_prompt = f"You are a helpful assistant. 请非常{emo_prompt}地说一句话。<|endofprompt|>"
            else:
                tts_instruct_prompt = f"You are a helpful assistant. Please say the sentence in a very {emotion} mood.<|endofprompt|>"

            tts_text = tts_data[f"tts_text_{target_language_code}"]
            normalized_text, word_count = get_word_account(target_language_code, tts_text)
            prompt_config = lang_configs[target_language_code]
            stress_pred_prompt = gemini_stress_prediction_prompt.format(
                language_name=prompt_config['name'],
                language_code=target_language_code,
                language_specific_note=prompt_config['note'],
                transcription=normalized_text,
                word_count=word_count
            )

            # 2. Randomly select a speaker for this specific text-emotion pair
            speaker_info = random.choice(speaker_prompts)
            speaker_prompt = speaker_info['audio_path']

            # 3. Generation Loop
            while len(stressed_word_distribution) < args.sample_size_for_distribution:
                sample_id = len(stressed_word_distribution)
                
                # --- TTS Generation ---
                chunks = []
                for model_output in cosyvoice.inference_instruct2(
                    tts_text=tts_text,
                    instruct_text=tts_instruct_prompt,
                    prompt_wav=speaker_prompt, stream=False
                ):
                    chunks.append(model_output["tts_speech"])
                tts_speech = torch.cat(chunks, dim=1)
                
                audio_path = os.path.join(stressed_distribution_subdir, f'{emotion.lower()}_sample_{d_id}_{sample_id}.wav')
                torchaudio.save(audio_path, tts_speech, cosyvoice.sample_rate)

                # --- Emotion Validation ---
                rec_result = emo2vec_model.generate(audio_path, granularity="utterance")
                emo_key = '开心/happy' if emotion == "happy" else '难过/sad'
                emotion_idx = rec_result[0]['labels'].index(emo_key)
                emotion_score = rec_result[0]['scores'][emotion_idx]

                if emotion_score <= 0.9:
                    try_count += 1
                    if try_count >= 10: 
                        break
                    continue

                # --- Gemini Stress Prediction (Multi-pass) ---
                uploaded_audio = client.files.upload(file=audio_path)
                n_pass_predictions = []
                
                for _ in range(args.n_pass):
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.0-flash', # Use the latest stable flash model
                            contents=[uploaded_audio, stress_pred_prompt],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=stress_prediction_schema,
                                temperature=1.0
                            ),
                        )
                        res_data = json.loads(response.text)
                        n_pass_predictions.append(res_data["stress_pattern"])
                    except Exception as e:
                        print(f"Gemini API Error on GPU {gpu_id}: {e}")
                        continue

                # --- Save Sample Info ---
                stressed_word_distribution.append(n_pass_predictions)
                try_count = 0
                sample_info = {
                    "speaker_prompt": speaker_prompt,
                    "audio_path": audio_path,
                    "transcription": tts_text,
                    "pred_stress_pattern": n_pass_predictions,
                    "emotion_score": float(emotion_score),
                }
                with open(audio_path.replace('.wav', '_info.json'), 'w') as f:
                    json.dump(sample_info, f, indent=4, ensure_ascii=False)

            # 4. Final Distribution Calculation for this text-emotion pair
            if len(stressed_word_distribution) == args.sample_size_for_distribution:
                # Convert list of lists of lists to a frequency map
                # (Expected shape: [Samples, Passes, WordCount])
                dist_array = np.array(stressed_word_distribution) 
                # Collapse passes (mean) then collapse samples (mean)
                freq = np.mean(dist_array, axis=(0, 1)) 
                
                dist_info = {
                    "emotion": emotion,
                    "tts_text": tts_text,
                    "stressed_word_distribution": freq.tolist()
                }
                out_path = os.path.join(args.stress_distribution_output_dir, f'{emotion.lower()}_{target_language_code}', f'{d_id}_dist.json')
                with open(out_path, 'w') as f:
                    json.dump(dist_info, f, indent=4)
        print(f"GPU {gpu_id} finished processing text index {d_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using OpenAI API")
    parser.add_argument(
        "--num_data", 
        type=int, 
        default=20, 
        help="Number of texts to generate")

    parser.add_argument(
        "--audio_dir", 
        type=str, 
        default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/audio_emo2stress1_direct", 
        help="Path to save generated audio")

    parser.add_argument(
        "--text_dir", 
        type=str, 
        default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/text_gemini_emo_aware",
        help="Directory containing text files for speech generation")

    parser.add_argument(
        "--libritts_metadata_json",
        type=str,
        default="/data/user_data/willw2/data/LibriTTS_R/libritts_r_testset_metadata.json",
        help="Path to the LibriTTS metadata JSON file, this is used as speaker prompts for english stress word distribution estimation")

    parser.add_argument(
        "--wenetspeech_metadata_json",
        type=str,
        default="/data/user_data/willw2/data/WenetSpeech4TTS/WenetSpeech4TTS_Premium_0/wenetspeech_premium_01_metadata.json",
        help="Path to the WenetSpeech metadata JSON file, this is used as speaker prompts for chinese stress word distribution estimation")

    parser.add_argument(
        "--sample_size_for_distribution",
        type=int,
        default=100,
        help="Number of samples to analyze from LibriTTS for stressed word distribution")

    parser.add_argument(
        "--stress_distribution_output_dir",
        type=str,
        default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/text_gemini_emo_aware_stress_distribution_gemini_pred",
        help="Directory to save the stressed word distribution results")

    parser.add_argument(
        "--n_pass", 
        type=int, 
        default=10,
        help="Number of passes for speech generation. Aggregate 10 results and vote for the highest possible stressed word from LLM stress words prediction")
    
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

    parser.add_argument(
        '--stress_threshold',
        type=float,
        default=0.7,
        help="The probability threshold for selecting stressed words from distribution"
    )

    parser.add_argument(
        "--target_language_code",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Target language code for TTS generation and stress distribution estimation (default: en)"
    )

    args = parser.parse_args()
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(args.text_dir, exist_ok=True)
    os.makedirs(args.stress_distribution_output_dir, exist_ok=True)
    # os.makedirs(args.target_speech_distribution_output_dir, exist_ok=True)

    # gen 1000 audios with libritts speaker prompts, with different emotions, and calculate the distribution of detected stressed words using whistress.
    # Generate TTS text using Gemini
    # get_stressed_word_distribution(args.libritts_metadata_json, args.sample_size_for_distribution, args.text_dir, args.stress_distribution_output_dir)

    # Translate to target language
    # translate_tts_text(args.text_dir, target_language="Chinese")

    # Get sentence-level base stress words distribution
    tts_jsons = glob.glob(os.path.join(args.text_dir, "*.json"))
    all_tts_data = []
    for i, tts_json in enumerate(tts_jsons):
        with open(tts_json, 'r', encoding='utf-8') as f:
            all_tts_data.append((i, json.load(f))) # Store as (original_index, data)
    all_tts_data = all_tts_data[:args.num_data] # Limit to num_data for quick testing, remove or adjust for full run
    # Multi-GPU Setup
    num_gpus = torch.cuda.device_count()
    # Split the 100 texts into chunks for each GPU
    chunks = [all_tts_data[i::num_gpus] for i in range(num_gpus)]
    processes = []
    mp.set_start_method('spawn', force=True) # Required for CUDA multiproc

    for i in range(num_gpus):
            p = mp.Process(
                target=worker, 
                args=(i, chunks[i], args, args.target_language_code)
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()



    # get_stressed_word_distribution_target_language(
    #     args.libritts_metadata_json, 
    #     args.wenetspeech_metadata_json,
    #     args.sample_size_for_distribution, 
    #     args.n_pass,
    #     args.text_dir,
    #     args.stress_distribution_output_dir, 
    #     # args.target_speech_distribution_output_dir, 
    #     target_language_code=args.target_language_code)

    # gen_speech_cosyvoice3(
    #     text_jsons=args.text_dir,
    #     config_type=args.config_type,
    #     output_dir=args.audio_dir,
    #     n_pass=args.n_pass
    # )