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
from whistress import WhiStressInferenceClient
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

from prompt import emo_speaker_prompt, SENTENCE_TYPES, Emo1Stress2, Emo2Stress1, Emo2Stress2, data_gen_config, sentence_domain_topic_data, OPENAI_KEY, GEMINI_KEY, tts_translation_prompt, stress_prediction_schema, translation_schema, gemini_stress_prediction_prompt, lang_configs

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

def get_word_account(lang_code, text):
    if lang_code == "en":
        text = re.findall(r"[\w']+", text)
        return text, len(text)
    elif lang_code == "zh":
        text = re.sub(r'[^\w\u4e00-\u9fff]+', '', text)
        return text, len(text)


def gen_text_gemini(num_data, domain_topic, config_type, emo1, emo2, text_dir):
    gemini_client = genai.Client(api_key=GEMINI_KEY)

    for i in range(num_data):
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

def translate_tts_text(target_folder: str, target_language: str):
    client = genai.Client(api_key=GEMINI_KEY)
    filepaths = glob.glob(os.path.join(target_folder, "*json"))
    print(f"Found {len(filepaths)} JSON files to process for translation in {target_folder}")
    
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        try:
            # Pass the natively built Schema to the config
            original_data[f"tts_text_en"] = original_data['original_text'] # keep the original English text for reference
            response = client.models.generate_content(
                model='gemini-3-flash-preview', 
                contents=tts_translation_prompt.format(target_language=target_language, tts_text=original_data['tts_text_en']),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=translation_schema,
                    temperature=0.1 
                ),
            )

            # The response is guaranteed to match the schema
            translated_result = json.loads(response.text)
            if target_language.lower() == "chinese":
                lang_code = 'zh'
            else:                
                lang_code = target_language.lower()

            original_data[f"tts_text_{lang_code}"] = translated_result["translated_tts_text"]
            print(f"  -> Translated text: {original_data[f'tts_text_{lang_code}']}")
            print(f"  -> Original text: {original_data['tts_text_en']}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(original_data, f, ensure_ascii=False, indent=4)
                
            print(f"  -> Successfully updated {filepath} in-place.")
        except Exception as e:
            print(f"  -> API Error processing {filepath}: {e}")


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
                                    temperature=0.1 # Low temperature for highly analytical/deterministic acoustic analysis
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
    cosyvoice = CosyVoiceAutoModel(model_dir=model_dir)
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
    parser.add_argument(
        "--num_data", 
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

    # parser.add_argument(
    #     "--target_speech_distribution_output_dir",
    #     type=str,
    #     default="/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstts/text_gemini_emo_aware_stress_distribution_zh",
    #     help="Directory to save the generated target language speech and stressed word distribution results")

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