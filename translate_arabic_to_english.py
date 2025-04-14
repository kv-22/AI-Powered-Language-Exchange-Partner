from flask import Flask, request, jsonify
import requests
import torch
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Load translation model
nmt_model_name = "Raghad-DD/opus-mt-ar-en-finetuned-ar-to-en-13"
tokenizer = MarianTokenizer.from_pretrained(nmt_model_name)
nmt_model = MarianMTModel.from_pretrained(nmt_model_name)

def translate_arabic_to_english(arabic_text):
    tokens = tokenizer(arabic_text, return_tensors="pt", padding=True, truncation=True)
    translated = nmt_model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

@app.route("/translation_arabic_to_english_audio", methods=["POST"])
def translate_audio_to_english_external_transcriber():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    files = {'file': (file.filename, file.stream, file.content_type)}

    try:
        # Call external transcription service
        response = requests.post("http://external-service/transcribe", files=files) #you should replace the URL in this line with correct one
        if response.status_code != 200:
            return jsonify({"error": "External transcription failed"}), 500

        data = response.json()
        if 'transcription' not in data:
            return jsonify({"error": "Transcription missing"}), 500

        arabic_text = data['transcription']
        english_text = translate_arabic_to_english(arabic_text)

        return jsonify({"translation": english_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500