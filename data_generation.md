# Data generation pipeline 

Install CosyVoice 3 and relevant packages: https://github.com/FunAudioLLM/CosyVoice


## Generate data distribution 
Use index-tts/data_processing/tts_data_synth/gen_data_in_distribution.py to generate texts and base stress pattern for texts.

## Generate final speech using CosyVoice 3
Use index-tts/data_processing/tts_data_synth/gen_speech.py to generate speech from estimated base stress patterns