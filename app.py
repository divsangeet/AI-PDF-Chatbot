from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from .env
load_dotenv()

# Create Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.7
)

print("🤖 AI Chatbot Started!")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        print("👋 Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    response = llm.invoke(question)

    # Print only the AI's answer
    if isinstance(response.content, list):
        print("AI:", response.content[0]["text"])
    else:
        print("AI:", response.content)

    print()