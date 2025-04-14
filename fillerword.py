from pydub import AudioSegment
from ollama import chat
from ollama import ChatResponse
from groq import Groq
import os
import whisper
import re

# define filler words
filler_words = {
    "right":0,
    "you know":0,
    "I mean":0,
    "I guess":0,
    "literally":0,
    "basically":0,
    "seriously":0,
    "kind of":0,
    "kinda":0,
    "just":0,
    "like":0,
    "um":0,
    "ah":0,
    "uh":0,
    "huh":0,
    'so':0,
    "and yeah":0,
    "okay so":0,
    "hmm":0
}

# convert to wav format
audio = AudioSegment.from_file("")
audio.export("output.wav", format="wav")

# detect filler words with whisper by setting initial_prompt
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio=file_path, word_timestamps=True, language='en', initial_prompt="I was like, I'm like, you know what I mean, kind of, kinda, um, ah, huh, and so, so um, uh, and um, like um, so like, like it's, it's like, I mean, yeah, ok so, uh so, so uh, yeah so, you know, it's uh, uh and, and uh, like")
    return result

result = transcribe_audio("output.wav")
print("Transcribed Text: ")
print(result["text"])
# print("\n")
# print(result['segments'])
# print("\n")

# print timestamps
# for segment in result['segments']:
#     print(''.join(f"{word['word']}[{word['start']}/{word['end']}]" 
#         for word in segment['words']))

# find count of each filler word
for key in filler_words.keys():
    filler_words[key] = len(re.findall(r'\b' + re.escape(key.lower()) + r'\b', result['text'].lower())) # \b considers it not as a substring

filtered_filler_words = {key: value for key, value in filler_words.items() if value > 0}

print("\nDetected filler words:")
for key, value in filtered_filler_words.items():
    print(f"{key}: {value}")
    
# calculate fwer
fwer = sum(filtered_filler_words.values()) / len(result['text'].split()) * 100
print(f"\nFiller Word Error Rate: {fwer:.2f}")
    
# optional but getting feedback from llm about filler words in input
filler_words = ", ".join(key for key in filtered_filler_words.keys())

prompt = f"""
Given the following input and the associated filler words, provide concise feedback on how to reduce filler word usage in the input.

Input: {result['text']}
Filler words: {filler_words}
"""

# print(prompt)

messages = [
  {
    'role': 'user',
    'content': prompt,
  }
]

# with groq
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama-3.3-70b-versatile" # "llama-3.3-70b-versatile or deepseek-r1-distill-llama-70b"
)

response = chat_completion.choices[0].message.content

# if using deepseek remove think tags
# response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

print(response)