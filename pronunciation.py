from groq import Groq
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ForCTC
from itertools import groupby
from jiwer import wer
import os
import librosa
import torch
from dotenv import load_dotenv
import os
import whisper
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library('C:\Program Files\eSpeak NG\libespeak-ng.dll')


# Load the .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("GROQ_API_KEY")

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# decode model output
def decode_phonemes(ids: torch.Tensor, processor: Wav2Vec2Processor, ignore_stress: bool = False) -> str:
    """CTC-like decoding. First removes consecutive duplicates, then removes special tokens."""
    # removes consecutive duplicates
    ids = [id_ for id_, _ in groupby(ids)]

    special_token_ids = processor.tokenizer.all_special_ids + [
        processor.tokenizer.word_delimiter_token_id
    ]
    # converts id to token, skipping special tokens
    phonemes = [processor.decode(id_) for id_ in ids if id_ not in special_token_ids]

    # joins phonemes
    prediction = " ".join(phonemes)

    # whether to ignore IPA stress marks
    if ignore_stress == True:
      prediction = prediction.replace("ˈ", "").replace("ˌ", "")

    return prediction

checkpoint = "bookbot/wav2vec2-ljspeech-gruut"

processor = Wav2Vec2Processor.from_pretrained(checkpoint)
model = Wav2Vec2ForCTC.from_pretrained(checkpoint)

# Make sure your audio is loaded at the model's expected sampling rate
sr = processor.feature_extractor.sampling_rate  # This will return 16000

# load user audio file to get phoneme
audio_array, _ = librosa.load("output.wav", sr=sr)

# Use Whisper to transcribe the audio
result = whisper_model.transcribe(audio_array)
transcribed_text = result['text']

print(f"\nDetected Speech using Whisper: {transcribed_text}")

def text_to_ipa(text):
    # Use phonemizer to get the IPA transcription
    ipa_transcription = phonemize(text, language='en-us',backend='espeak')
    return ipa_transcription

reference_phonemes = text_to_ipa(transcribed_text)
print("\nIPA Transcription:", reference_phonemes)

# Load user audio file to match the model's expected sampling rate
sr = processor.feature_extractor.sampling_rate
# You can also directly load the audio using librosa or another method

audio_array, _ = librosa.load("output.wav", sr=sr)

inputs = processor(audio_array, return_tensors="pt", padding=True)

# Inference with Wav2Vec2
with torch.no_grad():
    logits = model(**inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)
phoneme_transcription = processor.batch_decode(predicted_ids)[0]

print(f"Phoneme Transcription (using Wav2Vec2): {phoneme_transcription}")


import numpy as np

def levenshtein(ref, hyp):
    ref = ref.split()
    hyp = hyp.split()
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost # substitution
            )
    return d[len(ref)][len(hyp)], d[len(ref)][len(hyp)] / len(ref)


# Clean stress markers from reference and prediction
reference_phonemes = reference_phonemes.replace("ˈ", "").replace("ˌ", "")
phoneme_transcription = phoneme_transcription.replace("ˈ", "").replace("ˌ", "")

# Convert to character lists
ref_list = list(reference_phonemes.replace(" ", ""))
hyp_list = list(phoneme_transcription.replace(" ", ""))

# Levenshtein similarity computation
distance, ratio = levenshtein(" ".join(ref_list), " ".join(hyp_list))

print(f"Levenshtein Distance (character-level): {distance}")
print(f"Phoneme Sequence Similarity: {(1 - ratio):.2%}")


# Build LLM prompt for feedback on pronunciation
prompt = f"""
I recorded this sentence: "{transcribed_text}".

Phoneme transcription for this sentence was detected as: "{phoneme_transcription}".

Please analyze whether my pronunciation matches the recorded sentence.
Give feedback on any mispronounced or unclear words.
Avoid writing phonemes in the explanation.
"""

# Send the prompt to the Groq API for feedback
messages = [
    {
        "role": "user",
        "content": prompt
    }
]

# Initialize Groq client and request feedback
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama-3.3-70b-versatile"
)

response = chat_completion.choices[0].message.content

print(f"Groq Feedback: {response}")

