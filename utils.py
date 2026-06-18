import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# .env file se API key load karne ke liye
load_dotenv()

def process_pdf(pdf_path: str):
    """
    Step 1 & 2: PDF load karna aur uske chhote chunks (pieces) banana.
    """
    # PyPDFLoader ka use karke PDF read karna
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Text ko chunks me split karna (Chunk size: 1000 characters, Overlap: 200 characters)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """
    Step 3: Chunks ko Google Gemini Embeddings me convert karke FAISS DB me save karna.
    """
    # Gemini embeddings model initialize karna
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # FAISS Vector Store banana chunks aur embeddings ka use karke
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Vector store ko locally ek folder me save kar lena taaki baar-baar API call na karni pade
    vector_store.save_local("faiss_db")
    return vector_store

def load_vector_store():
    """
    Saved FAISS database ko wapas load karne ke liye helper function.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists("faiss_db"):
        return FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    return None