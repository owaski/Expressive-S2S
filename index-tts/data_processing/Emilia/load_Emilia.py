import os
import sys
import json
import glob
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_emilia_dataset():
    path = "Emilia/EN/*.tar"
    # Use streaming mode to avoid loading everything into memory
    dataset = load_dataset(
        "amphion/Emilia-Dataset", 
        data_files={"en": path}, 
        split="en",
    )



def check_statistics():
    metadata_jsons = glob.glob("/data/user_data/willw2/data/EMILIA/EN-B000000/diarized_metadata/*.json")

    output_json_path = "/data/user_data/willw2/data/EMILIA/emilia_stressed_transcripted_audios_metadata.json"
    print(f"Found {len(metadata_jsons)} metadata files.")

    durations = []
    stressed_transcripted_audios = []
    for metadata_path in tqdm(metadata_jsons):
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "stressed_words_transcription" in data.keys():
                stressed_transcripted_audios.append(data)
                data['audio_file'] = metadata_path.replace("diarized_metadata", "diarized_audio").replace("json", "mp3")
                durations.append(data["duration_sec"])
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(stressed_transcripted_audios, f, indent=4)
    print(f"Saved {len(stressed_transcripted_audios)} stressed and transcripted audio metadata to {output_json_path}")
    
    # histogram of durations
    plt.hist(durations, bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Audio Durations in Emilia Dataset")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of Audio Files")
    plt.grid(True)
    plt.savefig("emilia_duration_distribution.png")
    plt.show()

    print(f"total duration in hours : {sum(durations)/3600.0}")

def check_duration():
    pass

if __name__ == "__main__":
    # ===== Load Emilia Dataset =====
    # load_emilia_dataset()

    # ===== Check Emilia Dataset Statistics =====
    check_statistics()

    # output_json_path = "/data/user_data/willw2/data/EMILIA/emilia_stressed_transcripted_audios_metadata.json"
    # with open(output_json_path, "r", encoding="utf-8") as f:
    #     stressed_transcripted_audios = json.load(f)
    
    # for entry in stressed_transcripted_audios:
    #     entry['audio_file'] = entry['audio_file'].replace("wav", "mp3")
    
    # with open(output_json_path, "w", encoding="utf-8") as f:
    #     json.dump(stressed_transcripted_audios, f, indent=4)
    # print(f"Updated audio file paths in {output_json_path}")


    