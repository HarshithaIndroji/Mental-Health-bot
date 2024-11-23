#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import random
import gradio as gr
from rapidfuzz import fuzz
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# In[3]:


with open("C:/Users/SAIHARSHITHA/OneDrive/Desktop/dataset [MConverter.eu].json", 'r', encoding='utf-8') as file:
    intents = json.load(file)["intents"]

# Load context file with specified encoding
with open("C:/Users/SAIHARSHITHA/OneDrive/Desktop/company xyz.txt", "r", encoding='utf-8') as file:
    context = file.read()


# In[4]:


tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


# In[7]:


conversation_log = []
q_table = {}  # For Q-Learning
alpha = 0.1  # Learning rate
gamma = 0.9  


# In[9]:


def preprocess_input(text):
    return text.lower().strip()


# In[10]:


def get_intent(user_input):
    best_match = None
    highest_score = 0
    for intent in intents:
        for pattern in intent["patterns"]:
            score = fuzz.ratio(user_input, pattern)
            if score > highest_score:
                highest_score = score
                best_match = intent
    return best_match if highest_score >= 50 else None


# In[11]:


def answer_question(question, context, max_tokens=512):
    # Split context into manageable chunks
    context_segments = [context[i:i+max_tokens] for i in range(0, len(context), max_tokens)]
    best_answer = ""
    best_score = -float("inf")
    
    for segment in context_segments:
        inputs = tokenizer(question, segment, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        start_index = torch.argmax(outputs.start_logits)
        end_index = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.decode(inputs['input_ids'][0][start_index:end_index])

        # Score answers based on model's confidence
        score = outputs.start_logits[0][start_index].item() + outputs.end_logits[0][end_index-1].item()
        if score > best_score:
            best_answer = answer
            best_score = score
    
    return best_answer


# In[12]:


def update_q_table(state, action, reward, next_state):
    global q_table
    # Initialize Q-Values if not present
    if state not in q_table:
        q_table[state] = {}
    if action not in q_table[state]:
        q_table[state][action] = 0
    
    # Get max Q-value for the next state
    max_next_q = max(q_table.get(next_state, {}).values(), default=0)
    
    # Update Q-Value using the Q-Learning formula
    q_table[state][action] += alpha * (reward + gamma * max_next_q - q_table[state][action])


# In[13]:


def chatbot_response(user_input, feedback=None):
    global q_table, conversation_log
    
    user_input = preprocess_input(user_input)

    if user_input == "bye":
        with open("conversation_log.json", "w") as log_file:
            json.dump(conversation_log, log_file)
        with open("q_table.json", "w") as q_file:
            json.dump(q_table, q_file)
        conversation_log = []
        return "Goodbye! Have a nice day 😊\nThank you for using Aneka. Your feedback helps improve my responses!"

    conversation_log.append({"user": user_input})
    
    matched_intent = get_intent(user_input)
    if matched_intent:
        action = random.choice(matched_intent["responses"])
    else:
        action = answer_question(user_input, context)
    
    # Reward mechanism based on feedback
    reward = feedback - 3 if feedback else 0  # Neutral feedback is 3
    
    # Update Q-Table
    state = user_input
    next_state = ""  # Assuming no sequential dependency here
    update_q_table(state, action, reward, next_state)
    
    conversation_log.append({"bot": action})
    return action

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("# Aneka - Mental Health Bot with Reinforcement Learning")
    gr.Markdown("Talk to me if you need any help!")
    
    with gr.Row():
        user_input = gr.Textbox(lines=2, placeholder="Type your question here...")
        feedback_slider = gr.Slider(1, 5, label="Feedback on Chat Quality", value=3)
    
    chat_history = gr.Textbox(lines=20, placeholder="Chat history...", value="", interactive=False)
    send_button = gr.Button("Send")

    def update_chat(user_input, feedback):
        response = chatbot_response(user_input, feedback if feedback is not None else None)
        if chat_history.value is None:
            chat_history.value = ""
        chat_history.value += f"\nUser: {user_input}\nAneka: {response}\n"
        return chat_history.value, "", 3

    send_button.click(
        update_chat,
        inputs=[user_input, feedback_slider],
        outputs=[chat_history, user_input]
    )

interface.launch()

