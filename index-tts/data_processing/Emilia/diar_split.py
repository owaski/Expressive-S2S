import os
import sys
import json
import glob
import torch
import torch.multiprocessing as mp
from pydub import AudioSegment
from tqdm import tqdm


from nemo.collections.asr.models import SortformerEncLabelModel
import nemo.collections.asr as nemo_asr

EMILIA_SR = 24000

def load_sortformer_model(device):
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1", map_location=device)
    diar_model.eval()
    diar_model = diar_model.to(device)
    print(f"the device of the diar model is {diar_model.device}")
    return diar_model

def load_asr_model(device):
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3", map_location=device)
    asr_model.eval()
    asr_model = asr_model.to(device)
    print(f"the device of the ASR model is {asr_model.device}")
    return asr_model

def has_more_than_one_speaker(segment_parts):
    if not segment_parts:
        return False
    if len(segment_parts) == 1:
        return False
    # segment_parts = [part.split(" ") for part in segment_parts]
    spk = segment_parts[0][-1]
    for segment_part in segment_parts[1:]:
        if segment_part[-1] != spk:
            return True
    return False

def sec_to_ms(t: float) -> int:
    # robust conversion for float timestamps (e.g., 2.8000000000000003)
    return int(round(float(t) * 1000))

def process_shards(rank, world_size, dataset_shards):
    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    
    print(f"Worker {rank} using device {device}")

    my_shards = dataset_shards[rank::world_size]
    print(f"Worker {rank} processing {len(my_shards)} shards.")

    # init Nvidia Sortformer diarization model
    diar_model = load_sortformer_model(device)
    asr_model = load_asr_model(device)

    # run speaker diarization and split longer audios
    for dataset_shard in my_shards:
        print(f"Processing dataset shard: {dataset_shard}")
        
        # create output dirs
        splited_audio_dir = os.path.join(dataset_shard, "diarized_audio")
        os.makedirs(splited_audio_dir, exist_ok=True)   
        splited_metadata_dir = os.path.join(dataset_shard, "diarized_metadata")
        os.makedirs(splited_metadata_dir, exist_ok=True)

        # collect audio files, process 1 audio file for testing
        audio_paths = glob.glob(os.path.join(dataset_shard, "audio", "*.mp3"))
        # audio_paths = audio_paths[:5]
        # audio_paths = ["/data/user_data/willw2/data/EMILIA/EN-B000000/audio/EN_B00000_S00520_W000003.mp3"]
        print()
        print(f"Found {len(audio_paths)} audio files in shard.")

        for audio_path in tqdm(audio_paths, desc=f"Shard {dataset_shard}"):
            print(f"Processing audio file: {audio_path}")
            predicted_segments = diar_model.diarize(audio=audio_path, batch_size=1)
            print(f"From diar model, the predicted segments: {predicted_segments}")

            for predicted_segment in predicted_segments:
                segment_parts = [segment.split(" ") for segment in predicted_segment]
                if not segment_parts:
                    print(f"No segments detected in {audio_path}, skipping")
                    continue

                # if more than 1 speaker detected, skip this
                if has_more_than_one_speaker(segment_parts):
                    print(f"More than one speaker detected in {audio_path}, skipping this audio file")
                    continue
                else:
                    # if the segment is longer than 10 seconds, run ASR and split the segments to smaller parts about 4-7 seconds long
                    if float(segment_parts[-1][1]) > 10.0:
                        print(f"Predicted diarization segments {segment_parts}")
                        predicted_transcription = asr_model.transcribe(audio=[audio_path], timestamps=True, batch_size=1)

                        for word_info in predicted_transcription[0].timestamp["word"]:
                            print(f"Word: {word_info['word']}, start: {word_info['start']}, end: {word_info['end']}")
                        print()

                        segments = []
                        current_words = []
                        current_start = None
                        current_end = None

                        for word_info in predicted_transcription[0].timestamp["word"]:
                            w = word_info["word"]
                            word_start = float(word_info["start"])
                            word_end = float(word_info["end"])

                            # initialize a new segment if needed
                            if current_start is None:
                                current_start = word_start
                                current_end = word_end
                                current_words = [w]
                                continue

                            # if adding this word would push us past ~7s, close current segment first
                            if (word_end - current_start) > 7.0:
                                segment_text = " ".join(current_words)
                                segments.append((current_start, current_end, segment_text))
                                print(f"Created segment: start={current_start:.2f}, end={current_end:.2f}, text={segment_text}")

                                # start a new segment with this word
                                current_start = word_start
                                current_end = word_end
                                current_words = [w]
                            else:
                                # otherwise, keep adding to current segment
                                current_words.append(w)
                                current_end = word_end
                        # flush remaining words into the last segment (may be < 7s)
                        if current_words:
                            segment_text = " ".join(current_words)
                            segments.append((current_start, current_end, segment_text))
                            print(f"Created last segment: start={current_start:.2f}, end={current_end:.2f}, text={segment_text}")
                        audio = AudioSegment.from_file(audio_path)
                        
                        for i, (start_sec, end_sec, text) in enumerate(segments):
                            start_ms = sec_to_ms(start_sec)
                            end_ms = sec_to_ms(end_sec)
                            print(f"Final segment: start={start_sec:.2f}, end={end_sec:.2f}, text={text}")
                            # save the splited audio and metadata to the output dirs
                            splited_audio_path = audio_path.replace(".mp3", f"_part{i}.mp3").replace("audio", "diarized_audio")
                            # save the audio segment to splited_audio_path
                            chunk = audio[start_ms:end_ms]
                            chunk.export(
                                splited_audio_path,
                                format="mp3",
                                bitrate="192k"
                            )
                            splited_json_path = audio_path.replace(".mp3", f"_part{i}.json").replace("audio", "diarized_metadata")
                            duration_sec = (end_ms - start_ms) / 1000.0
                            segment_json = {
                                "segment_id": i,
                                "audio_file": os.path.basename(splited_audio_path),
                                "start_sec": float(start_sec),
                                "end_sec": float(end_sec),
                                "duration_sec": round(duration_sec, 3),
                                "text": text
                            }
                            with open(splited_json_path, "w", encoding="utf-8") as f:
                                json.dump(segment_json, f, ensure_ascii=False, indent=2)
                            print(f"Saved splited audio to {splited_audio_path} and metadata to {splited_json_path}")
                    else:
                        print(f"Segment is shorter than 10 seconds, no need to split: {segment_parts}")
                        # also save the corresonding audio and metadata to the output dirs
                        start_sec = float(segment_parts[0][0])
                        end_sec = float(segment_parts[-1][1])
                        start_ms = sec_to_ms(start_sec)
                        end_ms = sec_to_ms(end_sec)
                        splited_audio_path = audio_path.replace(".mp3", f"_part0.mp3").replace("audio", "diarized_audio")
                        # save the audio segment to splited_audio_path
                        audio = AudioSegment.from_file(audio_path)
                        chunk = audio[start_ms:end_ms]
                        chunk.export(
                            splited_audio_path,
                            format="mp3",
                            bitrate="192k"
                        )
                        splited_json_path = audio_path.replace(".mp3", f"_part0.json").replace("audio", "diarized_metadata")
                        duration_sec = (end_ms - start_ms) / 1000.0

                        original_json_path = audio_path.replace("audio", "metadata").replace(".mp3", ".json")
                        with open(original_json_path, "r", encoding="utf-8") as f:
                            original_data = json.load(f)
                            original_text = original_data.get("text", "")
                        segment_json = {
                            "segment_id": 0,
                            "audio_file": os.path.basename(splited_audio_path),
                            "start_sec": round(float(start_sec), 3),
                            "end_sec": round(float(end_sec), 3),
                            "duration_sec": round(duration_sec, 3),
                            "text": original_text
                        }
                        with open(splited_json_path, "w", encoding="utf-8") as f:
                            json.dump(segment_json, f, ensure_ascii=False, indent=2)
                        print(f"Saved splited audio to {splited_audio_path} and metadata to {splited_json_path}")

def main():
    # collect dataset shards
    dataset_dir = "/data/user_data/willw2/data/EMILIA"
    dataset_shards = glob.glob(os.path.join(dataset_dir, "EN-*"))
    dataset_shards.sort()
    print(f"Found {len(dataset_shards)} dataset shards.")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        mp.spawn(process_shards, args=(num_gpus, dataset_shards), nprocs=num_gpus, join=True)
    else:
        process_shards(0, 1, dataset_shards)

if __name__ == "__main__":
    main()