from groq import Groq
import os

user_input = input("\n: ")

# prompt = f"""
# Is the input grammatically correct?

# Input: {user_input}
# """

prompt = f"""
You are an expert in English grammar. You are given a learner’s input.
Based on the input, you need to identify grammatical errors in it.
If there are no errors, you SHOULD NOT suggest alternatives.
Give feedback to the learner on how they are doing.
You SHOULD NOT answer questions that provide information other than grammar.

Input: {user_input}
Response:
"""

# prompt = f"""
# You are a grammar checker for language learners. 
# Given the following text, determine if it contains grammatical errors. 
# If there are no grammatical errors, respond with "No errors found. You're doing great!" 
# If there are grammatical errors, explain what the errors are and provide a suggestion for correction.

# Here are some examples:

# Text: She go to the park.
# Response: She goes to the park.

# Text: I can speaks English.
# Response: I can speak English.

# Text: I read a book.
# Response: No errors found. You're doing great!

# Text: {user_input}
# Response:
# """

print(prompt)

messages = [
  {
    'role': 'user',
    'content': prompt,
  }
]

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama-3.3-70b-versatile"
)

response = chat_completion.choices[0].message.content

print(response)