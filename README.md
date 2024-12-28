# Mental-Health-bot Using Reinforcement Learning 
This project presents Aneka, an AI-driven mental health chatbot designed to provide accessible 
emotional support and guidance using Reinforcement Learning (RL) for dynamic response 
optimization.  
 
Recognizing the limitations of traditional mental health services—such as high costs, accessibility 
barriers, and stigma—Aneka leverages AI to offer a scalable alternative that learns and adapts based on 
user interactions, enhancing its ability to deliver compassionate and relevant responses.  
 
By analysing user feedback and conversational patterns, the RL component refines the bot’s responses, 
making interactions increasingly personalized and supportive over time. Aneka is structured around a 
diverse set of intents derived from real-life mental health conversations and utilizes contextual language 
understanding to address user concerns effectively. 
 
Ultimately, this project aims to bridge gaps in mental health accessibility by providing a responsive, 
empathetic digital companion that offers meaningful assistance to individuals seeking help, regardless 
of time or location. 
 
The bot also employs Natural Language Processing (NLP) techniques, utilizing intent recognition to 
match user statements with predefined patterns, ensuring accurate and relevant responses. In cases 
where intent is unclear, Aneka leverages a BERT-based QA model to provide context-based answers 
from a knowledge base.

THE STEP BY STEP EXAPLANTION FOR THIS PROJECT
1. Set Up Your Python Environment
Ensure Python is installed on your system.
Create a virtual environment (optional but recommended) for managing dependencies.
2. Install Required Libraries
Use pip to install the following libraries:

transformers: For working with the BERT model.
torch: For handling tensor operations and loading the model.
rapidfuzz: For fuzzy string matching to detect intent.
gradio: For building the chatbot interface.
json: For handling conversation logs (already included in Python).
3. Download Pre-trained Models
The code uses the BERT model, "bert-large-uncased-whole-word-masking-finetuned-squad".
When running the code, this model will automatically download using the transformers library if not already available.
4. Set Up the Context File
Ensure you have a context file or predefined context string to provide the model with information for answering user questions.
5. Define Intent Patterns
Prepare a JSON file or Python dictionary with intents, patterns, and responses. This helps the bot match user inputs to specific intents.
6. Code Breakdown
Preprocessing: The user input is converted to lowercase and trimmed for better matching.
Intent Matching: Fuzzy matching is performed to find the closest intent for the input.
QA Model: If no intent matches, the bot uses the BERT model to generate answers from the context file.
Feedback and Logging: User interactions and feedback are logged into JSON files for improving the bot later.
7. Launch the Gradio Interface
Interface Design: The Gradio library creates a web-based interface with a textbox for user input, a feedback slider, and a display for chat history.
Send Button: Connects the input and bot's response with the chat interface.
Conversation Logging: The conversation and feedback are saved when the user ends the chat by typing "bye."
8. Run the Code
Execute the script in an environment like Jupyter Notebook, VSCode, or directly from the command line using python script_name.py.
A browser window will open with the chatbot interface.
9. Interacting with the Bot
Type a message in the textbox and click the "Send" button.
View the bot’s responses and provide feedback using the slider.
10. Saving Logs and Improving
When the chat ends with "bye," the conversation history and feedback are saved to respective JSON files.
Use this data for further analysis or to improve intent matching.

PS: This is executed in Jupyter Notebook








