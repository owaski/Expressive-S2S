import os
import sys
import json
import re
# import Field
from pydantic import BaseModel
from google.genai import types

expresso_happy_json = "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/eval/expresso/happy_audio.json"
expresso_sad_json = "/data/user_data/willw2/course_project_repo/Expressive-S2S/data/eval/expresso/sad_audio.json"
emo_speaker_prompt = {}
with open(expresso_happy_json, "r", encoding="utf-8") as f:
    happy_audio_data = json.load(f)
    emo_speaker_prompt["happy"] = happy_audio_data

with open(expresso_sad_json, "r", encoding="utf-8") as f:
    sad_audio_data = json.load(f)
    emo_speaker_prompt["sad"] = sad_audio_data

with open('/data/user_data/willw2/course_project_repo/Expressive-S2S/index-tts/data_processing/tts_data_synth/sentence_domain_topic.json', 'r') as f:
    sentence_domain_topic_data = json.load(f)

# OPENAI_KEY = "YOUR KEY HERE"
OPENAI_KEY = None
GEMINI_KEY="AIzaSyDhidZLY2k1SUQlYr5JJBH42hi4W0tmgYs"

SENTENCE_TYPES = [
    "Statement",
    "Question",
    "Command",
    "Exclamation",
    'Request',
    'Suggestion',
    'Invitation',
    'Offer',
    'Opinion',
    'Warning'
]

tts_translation_prompt = """
    Translate the 'tts_text' from the following JSON into natural {target_language}.
    Only translate the text. Do not modify or analyze any other fields.
    
    Original text: "{tts_text}"
"""

class Emo1Stress2(BaseModel):
    """Configuration: Fixed Emotion, Varying Stress"""
    original_text: str
    emotion_1: str
    stressed_text_1: str
    intention_e1_s1: str
    stressed_text_2: str
    intention_e1_s2: str

class Emo2Stress1(BaseModel):
    """Configuration: Varying Emotion, Fixed Stress"""
    original_text: str
    emotion_1: str
    emotion_2: str
    stressed_text_1: str
    intention_e1_s1: str # Intention for Emo 1
    intention_e2_s1: str # Intention for Emo 2

class Emo2Stress2(BaseModel):
    """Configuration: Full 2x2 Factorial Matrix"""
    original_text: str
    emotion_1: str
    emotion_2: str
    stressed_text_1: str
    stressed_text_2: str
    # Map intentions clearly: (Emotion_Stress)
    intention_e1_s1: str 
    intention_e1_s2: str
    intention_e2_s1: str
    intention_e2_s2: str

data_gen_config = {
    "emo1stress2": {
        "prompt": 
                    """
                        # ROLE: Affective Prosody Analyst & Expert Linguist
                        # TASK: Generate data for the class `Emo1Stress2`.
                        # CONTEXT: Emotion (A): {emo_1}, Domain: {domain}, Topic: {topic}

                        # STRICT REQUIREMENTS:
                        1. Naturalness: Generate one natural English sentence (5-15 words) for the given domain.
                        2. Emotion: 'emotion_1' MUST be "{emo_1}".
                        3. Structure: Use exactly one pair of asterisks (*) per variation to highlight the stressed word.
                        4. Variable: Shift the focus by stressing DIFFERENT words in 'stressed_text_1' and 'stressed_text_2' using *asterisks*.
                        5. Intentions: Describe how emotions affects stressed word's vocal delivery (<30 words).
                        6. CosyVoice Optimization: Intentions must use EXTREME acoustic directives. Focus on: 'Aggressive pitch peak', 'Breathier texture', 'Vocal fry', 'Extended vowel duration', or 'Rapid staccato rhythm'.

                        # OUTPUT JSON:
                        {{
                            "original_text": "...",
                            "emotion_1": "{emo_1}",
                            "stressed_text_1": "...",
                            "intention_e1_s1": "[Direct vocal instruction for {emo_1} and stressed Word A focus]",
                            "stressed_text_2": "...",
                            "intention_e1_s2": "[Direct vocal instruction for {emo_1} and stressed Word B focus]"
                        }}
                    """,
        "return_class": Emo1Stress2,
        "tts_instruct_prompt": "You are a helpful assistant. Speak in a {emo} mood: {intention}"
    },

    "emo2stress1": {
        "prompt": 
                    """
                        # ROLE: Affective Prosody Analyst & Expert Linguist
                        # TASK: Generate data for the class `Emo2Stress1`.
                        # CONTEXT: Emotion (A): {emo_1}, Emotion (B): {emo_2}, Domain: {domain}, Topic: {topic}

                        # STRICT REQUIREMENTS:
                        1. Naturalness: Generate one natural English sentence (5-15 words).
                        2. Fixed Stress: 'stressed_text_1' and 'stressed_text_2' MUST use the EXACT SAME word with *asterisks*.
                        3. Variable: 'emotion_1' must be "{emo_1}" and 'emotion_2' must be "{emo_2}".
                        4. Intentions: Describe how emotions affects stressed word's vocal delivery (<30 words).
                        4. Intention Logic:
                            - intention_e1_s1: Describe how {emo_1} affects the stressed word's vocal delivery (<30 words).
                            - intention_e2_s1: Describe how {emo_2} affects the stressed word's vocal delivery (<30 words).
                        5. CosyVoice Optimization: Intentions must use EXTREME acoustic directives. Focus on: 'Aggressive pitch peak', 'Breathier texture', 'Vocal fry', 'Extended vowel duration', or 'Rapid staccato rhythm'.

                        # OUTPUT JSON:
                        {{
                            "original_text": "...",
                            "emotion_1": "{emo_1}",
                            "emotion_2": "{emo_2}",
                            "stressed_text_1": "...",
                            "intention_e1_s1": "[Direct vocal instruction for {emo_1} and the stressed word focus]",
                            "intention_e2_s1": "[Direct vocal instruction for {emo_2} and the stressed word focus]"
                        }}
                    """,
        "return_class": Emo2Stress1,
        "tts_instruct_prompt": "You are a helpful assistant. Speak in a {emo} mood: {intention}"
    },

    "emo2stress2": {
        "prompt": 
                    """
                        # ROLE: Voice Performance Architect & Expert Linguist
                        # TASK: Generate a 2x2 Factorial Matrix for `Emo2Stress2`.
                        # CONTEXT: Emotion (A): {emo_1}, Emotion (B): {emo_2}, Domain: {domain}, Topic: {topic}

                        # STRICT REQUIREMENTS:
                        1. Naturalness: Generate one natural English sentence (5-15 words).
                        2. Structure: Use exactly one pair of asterisks (*) per variation to highlight the stressed word.
                        3. Dimensions: Combine 2 Emotions ({emo_1}, {emo_2}) with 2 Stress positions (Word A, Word B).
                        4. Intention Logic:
                            - intention_e1_s1: Describe how {emo_1} affects Word A's vocal delivery (<30 words).
                            - intention_e1_s2: Describe how {emo_1} affects Word B's vocal delivery (<30 words).
                            - intention_e2_s1: Describe how {emo_2} affects Word A's vocal delivery (<30 words).
                            - intention_e2_s2: Describe how {emo_2} affects Word B's vocal delivery (<30 words).
                        5. CosyVoice Optimization: Intentions must use EXTREME acoustic directives. Focus on: 'Aggressive pitch peak', 'Breathier texture', 'Vocal fry', 'Extended vowel duration', or 'Rapid staccato rhythm'.

                        # Example output:
                        {{
                            "original_text": "...",
                            "emotion_1": "{emo_1}",
                            "emotion_2": "{emo_2}",
                            "stressed_text_1": "...",
                            "stressed_text_2": "...",
                            "intention_e1_s1": "[Direct vocal instruction for {emo_1} and Word A focus]",
                            "intention_e1_s2": "[Direct vocal instruction for {emo_1} and Word B focus]",
                            "intention_e2_s1": "[Direct vocal instruction for {emo_2} and Word A focus]",
                            "intention_e2_s2": "[Direct vocal instruction for {emo_2} and Word B focus]"
                        }}
                    """,
        "return_class": Emo2Stress2,
        "tts_instruct_prompt": "You are a helpful assistant. Speak in a {emo} mood: {intention}"
    }
}

class StressWordSelectionOutput(BaseModel):
    selected_word: str
    word_index: int
    reasoning: str

extra_stress_word_selection_prompt = """
    You are an expert in English phonetics, prosody, and emotional speech synthesis. 

    I am generating text-to-speech (TTS) audio with a "{Emotion}" emotion. 
    Here is the sentence: "{Sentence}"

    Currently, the TTS engine naturally places primary or secondary stress on these words (the underlying pattern): {Underlying_Pattern_List}

    Your task is to select exactly ONE new word from the sentence to receive deliberate phonetic stress. This new stress must sound natural for a human speaker conveying a "{Emotion}" emotion.

    RULES:
    1. You MUST NOT select a word from the underlying pattern list.
    2. You MUST NOT select function words (e.g., articles like "the", "a"; prepositions like "to", "from", "of"; auxiliary verbs like "was", "is"; or basic pronouns like "I", "you", "we"), unless shifting the stress to a pronoun completely changes the pragmatic meaning in a way that fits the emotion.
    3. You MUST select a content word (e.g., a noun, main verb, adjective, or adverb) that adds nuance, contrast, or intensity to the emotion.
    4. Output your choice in valid JSON format.

    Output Format:
    {
        "selected_word": "the word you chose",
        "word_index": [the 0-based index of the word in the sentence],
        "reasoning": "A brief, 1-sentence explanation of how stressing this word enhances the {Emotion} emotion."
    }
"""

lang_configs = {
    'zh': {
        'name': 'Mandarin Chinese',
        'note': 'Identify stress by character. Marked tone emphasis and vowel lengthening are primary cues.'
    },
    'en': {
        'name': 'English',
        'note': 'Identify stress by word. Look for pitch peaks and increased intensity on primary stressed syllables.'
    }
}

gemini_stress_prediction_prompt = """
    ### Task
    You are an expert phonetician specialized in {language_name} phonology. Analyze the provided audio waveform against the ground-truth transcription to identify intentional Lexical Stress and Pragmatic Prominence.

    ### Language Context
    - Language: {language_name} ({language_code})
    - Note: {language_specific_note}

    ### Input
    - Transcription: {transcription}
    - Unit Count: {word_count}

    ### Analysis Criteria: Structural Baseline vs. Intentional Focus
    Identify a unit as "stressed" (1) ONLY if its acoustic prominence deviates significantly from the expected structural baseline of a neutral sentence.

    GENERAL PRINCIPLES:
    1. **Structural Baseline Awareness:** Filter out natural structural artifacts. 
       - **Initial Pitch Reset:** The first word often has a higher pitch naturally (declination start). Do NOT mark it as stressed unless intensity and duration are also significantly higher than subsequent content words.
       - **Final Lengthening:** Words at the end of a sentence naturally elongate. Only mark as stressed if there is a distinct pitch focus or increased vocal effort.
    2. **Cue Clustering:** Prioritize units where multiple acoustic dimensions align (e.g., higher peak pitch + increased intensity + localized vowel lengthening).
    3. **Relative Comparison:** Evaluate each unit relative to its immediate neighbors. A unit is only stressed if it stands out as a "peak" in the local melodic and rhythmic contour.
    4. **Physical Effort:** Only report 1 if you detect "vocal weight"—increased breath pressure and articulatory precision—compared to surrounding units.

    ### Constraints
    - Output: A valid Python-style list of integers (0 or 1).
    - Length: Exactly {word_count}.
    - Do not provide explanations, prose, or markdown formatting outside the list.

    ### Output Format
    [0, 1, 0, ...]
"""

distribution_stress_word_selection_prompt = "You are a helpful assistant. Speak in a {emo} mood: {intention}"

stress_prediction_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "stress_pattern": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.INTEGER, # Specifies that the list contains integers (0 or 1)
            ),
            description="A list of binary values (0 or 1) indicating whether each word in the transcription is stressed. 0 means not stressed, 1 means stressed. The length of this list MUST exactly match the number of words in the transcription."
        )
    },
    required=["stress_pattern"] # Enforces that this key is always returned
)

translation_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "translated_tts_text": types.Schema(
            type=types.Type.STRING,
            description="The translated version of the original tts_text."
        )
    },
    # Ensure the model knows this field is mandatory
    required=["translated_tts_text"]
)

# system_prompt = """
#     You are an expert linguist and emotional intelligence analyst. 
#     Your task is to generate a single, natural English sentence (5-15 words) based on a specific Type, Domain, and Topic. You must demonstrate how shifting the vocal emphasis (indicated by *asterisks*) fundamentally alters both the speaker's intent and their underlying emotional state.
#     ### Strict Requirements:
#     1. Naturalness: The sentence must be something a person would actually say in the specified domain.
#     2. Structure: Use exactly one pair of asterisks (*) per variation to highlight the stressed word.
#     3. Intention: Describe the pragmatic goal of the stress (concise, <30 words).
#     4. Emotional Constraint: 
#         - Variation 1 MUST result in a **Happy** emotion (e.g., relief, joy, excitement).
#         - Variation 2 MUST result in a **Sad** emotion (e.g., disappointment, regret, loneliness).
#     5. Contrast: The two variations must provide a distinct shift in meaning or attitude.
# """

# prompt_template = """
#     Sentence Type: {}
#     Domain: {}
#     Topic: {}
#     Generate a response based on the requirements. Ensure that 'emotion_1' is Happy and 'emotion_2' is Sad.
    
#     Result Structure:
#     - Original Text: (The sentence without markers)
#     - Stressed Text 1 (Happy): (Sentence with *stress*)
#     - Intention 1: (Why they are happy)
#     - Emotion 1: happy
#     - Stressed Text 2 (Sad): (Sentence with *stress*)
#     - Intention 2: (Why they are sad)
#     - Emotion 2: sad
# """