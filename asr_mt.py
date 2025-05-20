def transcribe(audio, pipe_asr):
    transcript = pipe_asr(audio)['text']
    return transcript

def translate(text, pipe_mt):
    translation = pipe_mt(text)[0]["translation_text"]
    return translation

