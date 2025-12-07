import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
import argparse
import os
from jiwer import wer

torch.set_float32_matmul_precision("high")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def load_model(model_id="openai/whisper-large-v3"):
    
    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True,
    ).to(device)

    # Enable static cache and compile the forward pass
    model.generation_config.cache_implementation = "static"
    model.generation_config.max_new_tokens = 256
    model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
        language='en',
        # batch_size=8,
        return_timestamps=True,
    )

    return pipe

def run_asr(pipe, speech_files, batch_size=8):
    results = []
    for i in tqdm(range(0, len(speech_files)), desc="ASR Inference"):
        batch_inputs = speech_files[i]
        with sdpa_kernel(SDPBackend.MATH):
            result = pipe(batch_inputs)
        results.append(result)
    return results

def main(args):
    pipe = load_model()
    with open(args.wav_path_file, 'r') as f:
        lines = [line.strip(' \n') for line in f.readlines()]
    for _ in tqdm(range(2), desc="Warm-up step"):
        with sdpa_kernel(SDPBackend.MATH):
            _ = pipe(lines[0], generate_kwargs={"min_new_tokens": 256, "max_new_tokens": 256})
    results = run_asr(pipe, lines)
    with open(args.output_file, "a") as f:
        for res in results:
            f.write(res["text"].strip() + "\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--wav-path-file", type=str, required=True, help="Path to the list of wav files to transcribe.")
    argparser.add_argument("--output-file", type=str, required=True, help="Path to the output transcription file.")
    args = argparser.parse_args()
    main(args)
