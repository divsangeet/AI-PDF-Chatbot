import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.7
)

st.title("🤖 AI Chatbot")

# Create chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        ai_response = response.content[0]["text"]
    else:
        ai_response = response.content

    # Save AI response
    st.session_state.messages.append(
        {"role": "assistant", "content": ai_response}
    )

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)