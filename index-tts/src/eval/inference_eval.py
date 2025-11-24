import gc
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))
from indextts.infer_v2 import IndexTTS2
import torch
import torch.nn as nn
from torchinfo import summary
import argparse
import json
import re
'''
This script will be used to run inference using the stress_aware IndexTTS model.

Use IndexTTS2 from indextts.infer_v2
Load the finetuned GPT checkpoint from /data/user_data/willw2/expressive_s2st/index-tts/experiment/finetune_outputs_stress17k/checkpoints
This model contains other training artifacts, so only load the relevant modules into IndexTTS2 (such as the gpt module, text embedding, etc.)
Utilized stress words marked with the <*> </*> tokens for inference testing.
'''



def load_indextts2_with_finetuned_checkpoint(
    ckpt_path: str="/data/user_data/willw2/expressive_s2st/index-tts/experiment/finetune_outputs_stress17k/checkpoints/stress17k-epoch=31-val_loss=2.3590.ckpt",
    indextts2_config_path: str="/data/user_data/willw2/expressive_s2st/index-tts/checkpoints/config.yaml",
    indextts2_model_dir: str="/data/user_data/willw2/expressive_s2st/index-tts/checkpoints"
    ) -> IndexTTS2:
    """
    Load IndexTTS2 model with finetuned GPT checkpoint.
    
    Args:
        ckpt_path: Path to the finetuned checkpoint file.
    
    Returns:
        IndexTTS2 model with loaded finetuned weights.
    """
    # 1. Initialize IndexTTS2
    indextts2 = IndexTTS2(cfg_path=indextts2_config_path, model_dir=indextts2_model_dir, use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
    print(f"===== DEBUG =====, the device of IndexTTS2 model: {indextts2.device}")
    # print(f"===== DEBUG =====, the originsl IndexTTS2 model: {indextts2.gpt}")

    # 2. Load the checkpoint dict directly
    # ckpt_path = "/data/user_data/willw2/expressive_s2st/index-tts/experiment/finetune_outputs_stress17k/checkpoints/stress17k-epoch=31-val_loss=2.3590.ckpt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # examine the checkpoint keys.
    '''
    Checkpoint keys: 
        ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters', 
        'gpt_state_dict', 
        'text_embedding_state_dict', 
        'mel_embedding_state_dict', 
        'final_norm_state_dict', 
        'text_head_state_dict', 
        'mel_head_state_dict', 
        'mel_pos_embedding_state_dict', 
        'text_pos_embedding_state_dict']
    '''
    print("Checkpoint keys:", list(checkpoint.keys()))

    if "wte.weight" not in checkpoint["gpt_state_dict"] and "mel_embedding_state_dict" in checkpoint:
        checkpoint["gpt_state_dict"]["wte.weight"] = checkpoint["mel_embedding_state_dict"]["weight"]

    # replace the gpt module of UnifiedVoice in indextts2 with the finetuned gpt module from the checkpoint.
    indextts2.gpt.gpt.load_state_dict(checkpoint['gpt_state_dict'])

    # replace the text_embedding module of UnifiedVoice in indextts2 with the finetuned text_embedding module from the checkpoint.
    # UnifiedVoice model is indestts2.gpt
    weight = checkpoint['text_embedding_state_dict']['weight']
    num_embeddings, embedding_dim = weight.shape

    # Replace the embedding with the correct shape
    indextts2.gpt.text_embedding = nn.Embedding(num_embeddings, embedding_dim)
    indextts2.gpt.text_embedding.load_state_dict(checkpoint['text_embedding_state_dict'])
    indextts2.gpt.text_embedding.to(indextts2.device)
    indextts2.gpt.mel_embedding.load_state_dict(checkpoint['mel_embedding_state_dict'])

    # Replace the text_head with the correct shape
    text_head_weight = checkpoint['text_head_state_dict']['weight']
    text_head_bias = checkpoint['text_head_state_dict']['bias']
    out_features, in_features = text_head_weight.shape
    indextts2.gpt.text_head = nn.Linear(in_features, out_features)
    indextts2.gpt.text_head.load_state_dict(checkpoint['text_head_state_dict'])
    indextts2.gpt.text_head.to(indextts2.device)

    indextts2.gpt.text_pos_embedding.load_state_dict(checkpoint['text_pos_embedding_state_dict'])
    indextts2.gpt.mel_pos_embedding.load_state_dict(checkpoint['mel_pos_embedding_state_dict'])
    indextts2.gpt.final_norm.load_state_dict(checkpoint['final_norm_state_dict'])
    indextts2.gpt.text_head.load_state_dict(checkpoint['text_head_state_dict'])
    indextts2.gpt.mel_head.load_state_dict(checkpoint['mel_head_state_dict'])
    return indextts2

def infer_example(text_txt_path, output_wav_path):
    with open(text_txt_path, 'r') as f:
        samples = f.read().strip()
    
    for i, text in enumerate(samples.split('\n')):
        indextts2_ft.infer(stress_control=True, spk_audio_prompt='examples/voice_07.wav', text=text, output_path=f"/data/user_data/willw2/data/expressive_t2s/generated_samples/finetune_infer_{i}.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)
        indextts2.infer(stress_control=False, spk_audio_prompt='examples/voice_07.wav', text=text, output_path=f"/data/user_data/willw2/data/expressive_t2s/generated_samples/base_infer_{i}.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)

    pass

def convert_stress_labels_stresstest(text):
    """
    Convert stress labels from Stresstest format to IndexTTS2 format.
    
    Args:
        text: Original text with stress labels (e.g., "LEONARDO PAINTED A REMARKABLE FRESCO.")
        
    Returns:
        Text with stress control markers inserted (e.g., "<*>LEONARDO</*> PAINTED A ...")
    """
    # For this example, let's assume the stress labels are indicated by uppercase words.
    words = text.split()
    stress_positions = []
    
    for idx, word in enumerate(words):
        if word.isupper():  # Assuming uppercase words are stressed
            stress_positions.append((idx, idx))
    
    stressed_text_tokens = insert_stress_tokens_preserving_positions(text, stress_positions, indextts2.tokenizer)
    stressed_text = ' '.join(stressed_text_tokens)
    
    return stressed_text

def get_stresstest_text(json_path):
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    texts = []
    for item in data:
        text = item['intonation']

        # convert "*I did not* steal this car."" to "<*>I did not</*> steal this car."
        text = re.sub(r'\*([^*]+)\*', r'<*>\1</*>', text)
        texts.append(text)
        print(f"converted text: {text}")

    return texts


def check_stress_functions(text: str, stress_positions: list, tokenizer):
    """
    Helper function to check the three stress-related functions.
    
    Args:
        text: Original text without stress markers
        stress_positions: List of (start_word_idx, end_word_idx) tuples
        tokenizer: The tokenizer to use
    
    Returns:
        bool: True if all checks pass, False otherwise
    """
    print(f"Testing with text: '{text}'")
    print(f"Stress positions: {stress_positions}")
    
    # Step 1: Insert stress markers
    marked_tokens = insert_stress_tokens_preserving_positions(text, stress_positions, tokenizer)
    print(f"Marked tokens: {marked_tokens}")
    
    # Convert tokens back to text
    marked_text = tokenizer.convert_tokens_to_string(marked_tokens)
    print(f"Marked text: '{marked_text}'")
    
    # Step 2: Extract stress positions from marked text
    extracted_positions = get_stress_word_indices(marked_text)
    print(f"Extracted positions: {extracted_positions}")
    
    # Check if positions match
    positions_match = extracted_positions == stress_positions
    print(f"Positions match: {positions_match}")
    
    # Step 3: Remove stress markers
    cleaned_text = remove_stress_control_markers(marked_text)
    print(f"Cleaned text: '{cleaned_text}'")
    
    # Check if text matches original
    text_match = cleaned_text == text
    print(f"Text matches original: {text_match}")
    
    overall_success = positions_match and text_match
    print(f"Overall test result: {'PASS' if overall_success else 'FAIL'}")
    print("-" * 50)
    
    return overall_success


if __name__ == "__main__":
    # ckpt_path = "/data/user_data/willw2/course_project_repo/Expressive-S2S/experiment/finetune_outputs_stress17k/checkpoints/stress17k-epoch=30-val_loss=2.5772.ckpt"
    # use argparse to get the checkpoint path from command line
    parser = argparse.ArgumentParser(description="IndexTTS2 Inference with Stress-Aware Finetuned Model")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the finetuned checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--original_ckpt_dir", type=str, default="examples/stresstest_examples.json", help="Path to the JSON file containing stresstest examples")
    args = parser.parse_args()

    samples = get_stresstest_text("/data/user_data/willw2/course_project_repo/Expressive-S2S/data/stresstest/stresstest.jsonl")
    # samples = ["Why did you give her <*>money</*>?"]

    indextts2_ft = load_indextts2_with_finetuned_checkpoint(ckpt_path=args.ckpt_path, indextts2_config_path=args.config_path, indextts2_model_dir=args.original_ckpt_dir)
    indextts2 = IndexTTS2(cfg_path=args.config_path, model_dir=args.original_ckpt_dir, use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
    


    # tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
    # text = "<*>leonardo</*> painted a remarkable fresco"
    # text = "I <*>hate</*> this"
    for i, text in enumerate(samples[:50]):
        indextts2_ft.infer(stress_control=True, spk_audio_prompt='index-tts/examples/voice_07.wav', text=text, output_path=f"data/generated_samples/finetune_infer_{i}.wav", verbose=True)
        indextts2.infer(stress_control=False, spk_audio_prompt='index-tts/examples/voice_07.wav', text=text, output_path=f"data/generated_samples/base_infer_{i}.wav", verbose=True)