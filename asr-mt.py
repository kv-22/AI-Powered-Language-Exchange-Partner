from transformers import pipeline
from pydub import AudioSegment

# audio = AudioSegment.from_file("")
# audio.export("output.wav", format="wav")

def transcribe(audio):
    pipe_asr = pipeline("automatic-speech-recognition", model="itskavya/whisper-large-informal-arabic-base")
    transcript = pipe_asr(audio)['text']
    return transcript

transcript = transcribe() 
print(transcript)

def translate(text):
    transcript = transcribe() # send wav file
    print(transcript)
    pipe_mt = pipeline("translation", model="itskavya/opus-mt-ar-en-finetuned-ar-to-en-final2")
    translation = pipe_mt(text)[0]["translation_text"]
    return translation

translation = translate(text=transcript) 
print(translation)


