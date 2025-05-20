import gradio as gr
from asr_mt import transcribe, translate
from content import get_topic, get_score, get_content_feedback
from pronunciation import get_text, get_ref, get_per, get_pronunciation_feedback
from fillerword import get_fwer, get_filler_word_feedback
from chatbot import chat_with_llm
from load_models import load
import uuid

def transcribe_audio_filler_word(audio):
    result = whisper_model.transcribe(audio=audio, word_timestamps=True, language='en', initial_prompt="I was like, I'm like, you know what I mean, kind of, kinda, um, ah, huh, and so, so um, uh, and um, like um, so like, like it's, it's like, I mean, yeah, ok so, uh so, so uh, yeah so, you know, it's uh, uh and, and uh, like")
    return result["text"]

def transcribe_audio(audio):
    result = whisper_model.transcribe(audio=audio, language='en')
    return result['text']

def transcribe_and_chat(audio, history, mode, topic, reading_text, session_id):
    print(audio) # shows it is a wav file
    print("s: ", session_id)
    
    if audio is None: # if empty audio return as it is
        return "", history

    # route to appropriate chatbot logic
    if mode == "default":
        message = transcribe_audio(audio) # for grammar
        grammar_feedback = chat_with_llm(session_id, message) 
        
        message_filler_word = transcribe_audio_filler_word(audio) # for filler word
        fwer, filler_words = get_fwer(message_filler_word)
        
        if filler_words:
            filler_word_feedback = get_filler_word_feedback(message_filler_word, filler_words)
            response = f"Grammar Feedback:\n\n{grammar_feedback}\n\nFiller Word Feedback:\n\n{filler_word_feedback}\nFiller Words: {filler_words}\nScore: {fwer}"
        else:
            response = f"Grammar Feedback:\n\n{grammar_feedback}\n\nFiller Word Feedback:\nNo filler words detected, keep it up!"
    elif mode == "translate":
        transcript = transcribe(audio, pipe_asr) # use arabic whisper small
        response = translate(transcript, pipe_mt) # use marian 
    elif mode == "topic":
        message = transcribe_audio(audio)
        score = get_score(topic, message, embedding_model)
        content_feedback = get_content_feedback(topic, message)
        response = f"{content_feedback}\n\nScore: {score}" 
    elif mode == 'pronunciation':
        ref = get_ref()
        phonemes, per = get_per(phoneme_model, phoneme_processor, audio, ref)
        pronunciation_feedback = get_pronunciation_feedback(reading_text, phonemes)
        response = f"{pronunciation_feedback}\n\nScore: {per}" 

    history = history or []
    history.append({"role": "user", "content": "▶️"}) # using messages style as tuples is deprecated
    history.append({"role": "assistant", "content": response})
    # history.append(("▶️", response))
    return "", history # some syntax for first updating the display, second holds the memory

def set_topic_mode():
    topic = get_topic()
    return "topic", "Topic Mode", topic, []

def inject_topic(topic):
    # return [("💡", topic)]  # display topic as first message
    return [{"role": "user", "content": "💡"}, {"role": "assistant", "content": topic}]  # display topic as first message

def set_reading_mode():
    reading_text = get_text()
    return "pronunciation", "Pronunciation Mode", reading_text, []

def inject_reading_text(reading_text):
    # return [("💡", reading_text)] # display reading text as first message
    return [{"role": "user", "content": "💡"}, {"role": "assistant", "content": reading_text}]

def set_session_id():
    return "default", "Freestyle Mode", str(uuid.uuid4()), [] # for session id in freetsyle

with gr.Blocks() as interface:
    # session state to share data/var in a session
    mode = gr.State("default")  # store current mode
    topic_state = gr.State("")
    reading_state = gr.State("")
    session_id = gr.State("")
    
    gr.Markdown("### AI-Powered Language Exchange Partner") # title
    mode_display = gr.Markdown("Freestyle Mode") 
    
    chatbot = gr.Chatbot(type="messages")
    mic = gr.Audio(type="filepath", label="Record yourself...") # take audio input
    
    with gr.Row(): # create buttons for different modes
        default_btn = gr.Button("Freestyle")
        topic_btn = gr.Button("Topic")
        pronunciation_btn = gr.Button("Reading")
        translate_btn = gr.Button("Translate")
    
    mic.change(transcribe_and_chat, [mic, chatbot, mode, topic_state, reading_state, session_id], [chatbot, chatbot]) # when audio is sent (change event), call the func and give the inputs and last is for output

    # mode switching
    topic_btn.click(set_topic_mode, None, [mode, mode_display, topic_state, chatbot]).then(inject_topic, topic_state, chatbot) # call func to set mode as topic, none means no input, update and return the variables from func, then for updating chatbot
    default_btn.click(set_session_id, None, [mode, mode_display, session_id, chatbot]).then(lambda: None, None, chatbot) # new session id when mode changed, then for clearing chatbot
    pronunciation_btn.click(set_reading_mode, None, [mode, mode_display, reading_state, chatbot]).then(inject_reading_text, reading_state, chatbot)
    translate_btn.click(lambda: ("translate", "Translate Mode"), None, [mode, mode_display]).then(lambda: None, None, chatbot)
    
    interface.load(set_session_id, None, [mode, mode_display, session_id, chatbot]) # to have new session id each time page loads
      
if __name__ == "__main__":
    whisper_model, embedding_model, phoneme_model, phoneme_processor, pipe_asr, pipe_mt = load() # load models once in the beginning
    interface.launch()
