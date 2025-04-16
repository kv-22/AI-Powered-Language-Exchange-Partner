from groq import Groq
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor
from itertools import groupby
from jiwer import wer
import os
import librosa
import torch

# decode model output
def decode_phonemes(ids: torch.Tensor, processor: Wav2Vec2Processor, ignore_stress: bool = False) -> str:
    """CTC-like decoding. First removes consecutive duplicates, then removes special tokens."""
    # removes consecutive duplicates
    ids = [id_ for id_, _ in groupby(ids)] # ctc can repeat same phoneme because of alignment so remove duplicate

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

model = AutoModelForCTC.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)
sr = processor.feature_extractor.sampling_rate

# load user audio file to get phoneme
audio_array, _ = librosa.load("output.wav", sr=sr)

inputs = processor(audio_array, return_tensors="pt", padding=True) # extract audio features

with torch.no_grad():
    logits = model(inputs["input_values"]).logits

predicted_ids = torch.argmax(logits, dim=-1)
prediction = decode_phonemes(predicted_ids[0], processor, ignore_stress=True) # remove ipa stress marks 

phonemes = prediction
print(prediction)

text = "I work at PWC consulting firm"
ref = "ˈaɪ w ˈɚ k ˈæ t p ˈi d ˈʌ b ə l j ˌu s ˈi k ə n s ˈʌ l t ɪ ŋ f ˈɚ m"
ref = ref.replace("ˈ", "").replace("ˌ", "") # remove stress mark
print(ref)

per = round(wer(ref, prediction) * 100, 2) # get phoneme error rate
print(per)

# prompt = f"""
# Briefly explain how the following text should be pronounced to improve pronunciation. Avoid writing phonemes in the explanation.

# Text: {text}
# """

prompt = f"""
Given the following text and the spoken phonemes, provide feedback on the pronunciation.
For incorrect pronunciation, avoid writing phonemes in the feedback, instead, write how they should be pronounced in simple English.


Text: {text}
Phonemes: {phonemes}
"""

# prompt = f"""
# Given the following reference phonemes and the spoken phonemes, provide feedback on the pronunciation.
# For incorrect pronunciation, avoid writing phonemes in the feedback, instead, write how they should be pronounced in simple English.


# Reference phonemes: {ref}
# Spoken phonemes: {phonemes}
# """

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama-3.3-70b-versatile"
)

response = chat_completion.choices[0].message.content

print(response)