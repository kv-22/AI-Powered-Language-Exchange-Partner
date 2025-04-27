from groq import Groq
from sentence_transformers import SentenceTransformer
import os

client =  Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
def get_topic():
    user_preferences = ["fashion", "technology", "music"]
    user_preferences = ",".join(x for x in user_preferences)

    prompt = f"""
    Given the following user preferences, suggest one topic for the user to practice their speaking skills. Provide your response as {'topic'}.

    User preferences: {user_preferences}
    """

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]


    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile"
    )

    response = chat_completion.choices[0].message.content

    print(response)
    return response

def get_score(topic, reply, model):
    sentences = [topic, reply]
    embeddings = model.encode(sentences)
    similarities = model.similarity(embeddings, embeddings)
    score = round(similarities[0][1].item() * 100, 2)

    if score < 1: # negative similarity as in not related so make it 0
        score = 0
        
    print(score)
    return score

def get_content_feedback(topic, reply):
    prompt = f"""
    Given the following topic and reply, provide feedback on the content of the reply.

    Topic: {topic}
    Reply: {reply}
    """

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]


    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        max_completion_tokens=500
    )

    response = chat_completion.choices[0].message.content

    print(response)
    return response

# sample usage
# topic = get_topic()
# reply = "" # add sample reply
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# score = get_score(topic, reply, embedding_model)
# feedback = get_content_feedback(topic, reply)
# print("Score: ", score)
# print("Feedback: ", feedback)
