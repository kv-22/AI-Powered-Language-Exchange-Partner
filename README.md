# AI-Powered-Language-Exchange-Partner

This project explores building an AI-Powered language exchange partner application to improve English speaking skills for native Arabic speakers. We fine-tune Whisper and Wave2Vec2-XLS-R to support informal Arabic ASR and Marian-MT to support translation of informal Arabic to English, so users can easily learn with their native language. The training notebooks can be found in the model_training folder. Furthermore, we evaluate the user on various factors such as English grammar, pronunciation, fluency, and content. Grammar and fluency are supported as free-style practice sessions, and pronunciation and content as predefined sessions. Whisper (base) is used for English transcription. For all evaluations, we use the Llama-3.3-70B LLM via Groq to provide qualitative feedback.

A Gradio app is developed as a prototype and can be easily tried by running the "app.py" file.

Additionally, the working of each factor can be found in the following files and tried by uncommenting the "sample usage":

- chatbot.py: for grammar checking assistant with LangChain
- filler_words.py: for fluency assessment based on predefined filler words
- pronunciation.py: for pronunciation assessment using Wave2Vec2
- content.py: for assessment based on a topic using all-MiniLM-L6-v2

## Prerequisite
To use the LLM with Groq:

- Create account from [here](https://console.groq.com/login) and generate an API key.

- Create a ".env" file in the project directory and put your api key in it as GROQ_API_KEY=xyz.

Also:

- Navigate to the project directory and download all dependencies by running "pip install -r requirements.txt" in the terminal or command prompt. 
