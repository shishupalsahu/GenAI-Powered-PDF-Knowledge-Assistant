import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from utils import process_pdf, create_vector_store, load_vector_store

# Page Config
st.set_page_config(
    page_title="GenAI-Powered PDF Knowledge Assistant",
    layout="wide"
)

st.title("🧠 BrainPDF")
st.subheader("Your Intelligent PDF Chat Assistant")
st.write("Upload any document and get instant answers powered by RAG.")

# Chat History Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("1. Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload your PDF here",
        type=["pdf"]
    )

    if uploaded_file is not None:

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("PDF Uploaded Successfully!")

        if st.button("Process & Index PDF"):

            with st.spinner("Processing PDF and creating vector database..."):

                chunks = process_pdf("temp.pdf")
                create_vector_store(chunks)

                st.success("Database Ready! You can now ask questions.")

                if os.path.exists("temp.pdf"):
                    os.remove("temp.pdf")

# Chat Section
# st.header("2. Chat with your Document")

# Show Previous Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask a question from the uploaded PDF")

if user_question:

    # Save User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    vector_store = load_vector_store()

    if vector_store is not None:

        docs = vector_store.similarity_search(
            user_question,
            k=4
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a professional assistant specialized in answering questions based only on the provided PDF context.

Context:
{context}

Question:
{user_question}

Instructions:
- Answer only from the provided context.
- If the answer is not available in the context, reply:
"The answer is not available in the given document."
- Keep the answer clear and accurate.
"""

        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

        with st.spinner("Analyzing document and generating answer..."):

            response = model.invoke(
                [HumanMessage(content=prompt)]
            )

        answer = response.content

        # Save Assistant Response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

    else:
        st.warning(
            "Please upload and process a PDF from the sidebar first."
        )