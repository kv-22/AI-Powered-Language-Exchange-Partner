from ollama import chat
from ollama import ChatResponse

user_input = input("\n: ")

# prompt = f"""
# Is the input grammatically correct?

# Input: {user_input}
# """

prompt = f"""
You are a grammar checker for language learners. 
Given the following input, determine if there are any grammatical errors. 
If the input is correct, respond with "No errors found. You're doing great!" 
If there are any mistakes, explain what the error is and provide a suggestion for correction.

Here are some examples:

Input: She go to the park.
Response: She goes to the park.

Input: I can speaks English.
Response: I can speak English.

Input: I read a book.
Response: No errors found. You're doing great!

Input: {user_input}
"""

# print(prompt)

messages = [
  {
    'role': 'user',
    'content': prompt,
  }
]

response: ChatResponse = chat(model='llama3.2', messages=messages)
print(response.message.content)