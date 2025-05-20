from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForCTC
from transformers import pipeline
import whisper

def load():
    whisper_base = whisper.load_model("base")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    phoneme_checkpoint = "bookbot/wav2vec2-ljspeech-gruut"
    phoneme_model = AutoModelForCTC.from_pretrained(phoneme_checkpoint)
    phoneme_processor = AutoProcessor.from_pretrained(phoneme_checkpoint)
    pipe_asr = pipeline("automatic-speech-recognition", model="itskavya/whisper-small-informal-arabic-base-aug")
    pipe_mt = pipeline("translation", model="itskavya/opus-mt-ar-en-finetuned-ar-to-en-final2")
    return whisper_base, embedding_model, phoneme_model, phoneme_processor, pipe_asr, pipe_mt