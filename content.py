from transformers import BertTokenizer, BertModel
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import torch

user_preferences = ["fashion", "technology"]
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

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama-3.3-70b-versatile"
)

response = chat_completion.choices[0].message.content

print(response)

topic = response
reply = ""

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
encoded_input_topic = tokenizer(topic, return_tensors='pt')
encoded_input_reply = tokenizer(reply, return_tensors='pt')
# print(encoded_input_topic)
# print(encoded_input_topic['input_ids'])
# print(encoded_input_topic['attention_mask'])
# print(encoded_input_topic['token_type_ids'])
# print(tokenizer.convert_ids_to_tokens(encoded_input_topic['input_ids'][0]))
# print(len(tokenizer.convert_ids_to_tokens(encoded_input_topic['input_ids'][0])))

with torch.no_grad():
    output_topic = model(**encoded_input_topic)
    output_reply = model(**encoded_input_reply)
    # print(output_topic)

embedding_topic = output_topic.last_hidden_state # last hidden state has the embeddings
embedding_reply = output_reply.last_hidden_state
# print(embedding_topic)
# print(embedding_topic.shape) # batch, tokens, dim (each token is represented by a 768 dim vector)

# to get embedding of the text, average embeddings of each token 
topic_embedding = torch.mean(embedding_topic, dim=1).numpy()
reply_embedding = torch.mean(embedding_reply, dim=1).numpy()
cos_similarity = cosine_similarity(topic_embedding, reply_embedding)
print(cos_similarity[0][0])

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

# client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# chat_completion = client.chat.completions.create(
#     messages=messages,
#     model="llama-3.3-70b-versatile",
#     max_completion_tokens=500
# )

# response = chat_completion.choices[0].message.content

# print(response)