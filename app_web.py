import os

import faiss
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Page configuration
st.set_page_config(
    page_title="AI PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide"
)


# Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

if not api_key:
    st.error("Google Gemini API key is not configured.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.7
)


# Embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# Extract text from PDF
def extract_pdf_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# Split text into chunks
def create_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_text(text)


# Create FAISS vector store
def create_vector_store(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# Search relevant document chunks
def search_documents(question, chunks, index, top_k=4):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for idx in indices[0]:

        if idx != -1:
            results.append(chunks[idx])

    return results


# Generate answer
def generate_answer(question, context):

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context.

If the answer cannot be found in the document, say:

"I couldn't find the answer in the uploaded PDF."

Do not invent information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        return response.content[0]["text"]

    return response.content


# User interface
st.title("📄 AI-Powered PDF Q&A Chatbot")

st.write(
    "Upload a PDF and ask questions about its contents "
    "using Retrieval-Augmented Generation (RAG)."
)


# PDF upload
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


# Process uploaded PDF
if uploaded_file:

    with st.spinner("Reading and processing your PDF..."):

        pdf_text = extract_pdf_text(uploaded_file)

        if not pdf_text.strip():

            st.error(
                "No readable text was found in this PDF."
            )

            st.stop()

        chunks = create_chunks(pdf_text)

        vector_store = create_vector_store(chunks)

    st.success(
        f"PDF processed successfully! Created {len(chunks)} text chunks."
    )

    st.divider()


    # Chat history
    if "pdf_messages" not in st.session_state:

        st.session_state.pdf_messages = []


    for message in st.session_state.pdf_messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    # Question input
    question = st.chat_input(
        "Ask a question about your PDF..."
    )


    if question:

        # Show user question
        st.session_state.pdf_messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        # Search document
        with st.spinner("Searching the document..."):

            relevant_chunks = search_documents(
                question,
                chunks,
                vector_store
            )

            context = "\n\n".join(
                relevant_chunks
            )


        # Generate answer
        with st.spinner("Generating answer..."):

            answer = generate_answer(
                question,
                context
            )


        # Show answer
        st.session_state.pdf_messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):

            st.markdown(answer)

else:

    st.info(
        "👆 Upload a PDF above to start asking questions."
    )