from groq import Groq
import os
import re

text = "I went to the market this morning to buy some fresh fruit and vegetables"
phonemes = "aɪ n ɛ n t t ə ð ə m ɑ ɹ k ɪ t s ð ɪ s m ɔ ɹ n ɪ ŋ t ə b aɪ s ʌ m f ɹ ɛ ʃ f ɹ u t t æ n d v ɛ d͡ʒ t ə b ə l z"
ref = "aɪ w ɛ n t t ə ð ə m ɑ ɹ k ɪ t ð ɪ s m ɔ ɹ n ɪ ŋ t ə b aɪ s ʌ m f ɹ ɛ ʃ f ɹ u t æ n d v ɛ d͡ʒ t ə b ə l z"

prompt = f"""
Briefly explain how the following text should be pronounced to improve pronunciation. Avoid writing phonemes in the explanation.

Text: {text}
"""

# prompt = f"""
# Given the following text and the spoken phonemes, provide feedback on the pronunciation.
# For incorrect pronunciation, avoid writing phonemes in the feedback, instead, write how they should be pronounced in simple English.


# Text: {text}
# Phonemes: {phonemes}
# """

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
    model="llama-3.3-70b-versatile" # "llama-3.3-70b-versatile" or "deepseek-r1-distill-llama-70b"
)

response = chat_completion.choices[0].message.content

# if using deepseek remove think tags
# response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

print(response)