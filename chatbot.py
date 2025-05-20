import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import trim_messages, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq

# set llm using groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0,
)

# to trim conversation history and prevent it from exceeding llm token limit
trimmer = trim_messages(
    max_tokens=1024,
    strategy="last", # keep recent, trim old
    token_counter=llm,
    include_system=True, # keep system prompt
    allow_partial=False,
    start_on="human",
)

few_shot_examples = [
    {"input": "She go to the park.", "output": "She goes to the park."},
    {"input": "I did tried new food.", "output": "I tried new food."},
    {"input": "I read a book.", "output": "No errors found. You're doing great!"}
]

example_prompt = ChatPromptTemplate.from_messages(
    [   (
            "user",
            "{input}"
        ),    
        (
            "assistant",
            "{output}"
        ),

    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=few_shot_examples
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert in English grammar. You are given a learner’s input.
            The input is transcribed speech, hence you SHOULD NOT correct punctuation.
            Based on the input, you need to identify grammatical errors in it.
            If there are no errors, you SHOULD NOT suggest alternatives.
            Give feedback to the learner on how they are doing.
            You SHOULD NOT answer questions that provide information other than grammar.
            """,
        ),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# create graph that holds the entire workflow
workflow = StateGraph(state_schema=MessagesState) # state contains information that flows through the workflow, use in built messages state as the convo history is the only thing we update

# function that will call llm
def call_llm(state: MessagesState):
    # state["messages"][-1].content
    trimmed_messages = trimmer.invoke(state["messages"]) # trim history
    prompt = prompt_template.invoke({"messages": trimmed_messages}) # prepare prompt
    # print(prompt)
    response = llm.invoke(prompt) # call llm 
    return {"messages": response} # update state


# add model node with llm function in graph
workflow.add_node("model", call_llm)

# add edge from start to model in graph
workflow.add_edge(START, "model")

# to save a checkpoint and restart convo from it using sqlite db
connection = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)

# compile graph and add memory checkpointer to save convo history
memory = SqliteSaver(connection)
app = workflow.compile(checkpointer=memory)

def chat_with_llm(thread_id, transcript):
    print("t: ", thread_id)
    # set identity for conversation
    config = {"configurable": {"thread_id": thread_id}}
    
    input_messages = [HumanMessage(transcript)]
    
    output = app.invoke({"messages": input_messages}, config)
    print(output)
    response = output["messages"][-1].content
    
    return response

# sample usage
# chat_with_llm("1", "") # change thread id for new conversation, transcript should be user input

# sample usage as a chatbot
# while True:
#     user_input = input("\n: ")
#     if user_input == 'q':
#         break
    
#     input_messages = [HumanMessage(user_input)]
    
#     # for non streaming output
#     output = app.invoke({"messages": input_messages},  {"configurable": {"thread_id": 2}})
#     output["messages"][-1].pretty_print()
    