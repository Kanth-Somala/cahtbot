import os
import json
import datetime
import random
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk

# Configure SSL and NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents from a JSON file
file_path = os.path.abspath("./intents.json")
try:
    with open(file_path, "r", encoding="utf-8") as file:
        intents = json.load(file)["intents"]
except FileNotFoundError:
    st.error("The intents.json file is missing. Please add it to the project directory.")
    st.stop()
except json.JSONDecodeError as e:
    st.error(f"Error decoding JSON file: {e}")
    st.stop()

# Preprocess the data for training
tags = []
patterns = []
for intent in intents:
    tags.extend([intent["tag"]] * len(intent["patterns"]))
    patterns.extend(intent["patterns"])

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot response function
def chatbot_response(input_text):
    input_vector = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_vector)[0]
    for intent in intents:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to respond to that."

# Main Streamlit App
def main():
    st.title("Dynamic Chatbot with NLP")
    st.sidebar.header("Navigation")
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.radio("Go to", menu)

    # Persistent session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if choice == "Home":
        st.subheader("Start a Conversation")
        st.write("Type your message below to interact with the chatbot.")

        # Display chat history above the input field
        if st.session_state.chat_history:
            for entry in st.session_state.chat_history:
                st.markdown(f"**You:** {entry['user']}")
                st.markdown(f"**Bot:** {entry['bot']}")
                st.markdown("---")
        else:
            st.write("No chat yet. Start a conversation!")

        # Input field for user message
        user_input = st.text_input(
            "You:",
            key="user_input",
            placeholder="Type your message here...",
            on_change=lambda: st.session_state.submit_chat(),
        )

        # Function to handle chat submission
        def submit_chat():
            if st.session_state.user_input:
                response = chatbot_response(st.session_state.user_input)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Add to chat history
                st.session_state.chat_history.append(
                    {"user": st.session_state.user_input, "bot": response, "time": timestamp}
                )
                # Clear the input field
                st.session_state.user_input = ""

        # Add the submit function to the session state
        if "submit_chat" not in st.session_state:
            st.session_state.submit_chat = submit_chat


    elif choice == "Conversation History":
        st.subheader("Conversation History")
        if st.session_state.chat_history:
            for entry in reversed(st.session_state.chat_history):  # Display in reverse order (latest first)
                st.markdown(f"**You ({entry['time']}):** {entry['user']}")
                st.markdown(f"**Bot:** {entry['bot']}")
                st.markdown("---")
        else:
            st.write("No conversation history available.")

    elif choice == "About":
        st.subheader("About This Chatbot")
        st.write("""
        This chatbot is designed to understand and respond to user inputs dynamically using NLP.
        - **Intent Detection:** The chatbot identifies intents using a Logistic Regression model trained on labeled data.
        - **Interactive Interface:** Built using Streamlit for an engaging web-based interaction experience.
        - **Extensible Design:** Easily extendable with more intents, patterns, and responses.
        """)

# Run the application
if __name__ == '__main__':
    main()
