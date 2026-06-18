# 📚 GenAI-Powered PDF Knowledge Assistant

A powerful, production-ready Generative AI application that allows users to upload PDF documents and have intelligent conversations with them. Built using **Python**, **LangChain**, **Google Gemini Pro**, and **FAISS Vector Search**, this project implements the advanced **RAG (Retrieval-Augmented Generation)** pipeline.

---

## 🚀 Features

- **Instant PDF Ingestion:** Extracts text smoothly from multi-page PDFs using `PyPDFLoader`.
- **Smart Chunking:** Splits dense text into overlapping paragraphs (`RecursiveCharacterTextSplitter`) to retain contextual meaning.
- **Semantic Vector Search:** Converts text chunks into multi-dimensional embeddings using Google GenAI Embeddings and stores them in a lightweight, local **FAISS** vector database.
- **Context-Aware Q&A:** Uses **Gemini-1.5-Flash** via LangChain to deliver highly accurate answers based *only* on the uploaded document's context.
- **Hallucination Guard:** Custom prompt-engineering ensures the model cleanly says *"The answer is not available in the given document"* instead of making up wrong answers.
- **Sleek UI:** Clean and interactive chat interface built entirely with Streamlit.

---

## 🛠️ Tech Stack & Tools

- **Language:** Python
- **Orchestration:** LangChain
- **LLM API:** Google Gemini API (`gemini-2.5-flash`)
- **Embeddings:** Google GenAI Embeddings (`models/embedding-001`)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Frontend UI:** Streamlit
- **PDF Parser:** PyPDF

---

## 📦 Installation & Setup

Follow these simple steps to run the project locally:

### 1. Clone the Repository
```bash
git clone [https://github.com/shishupalsahu/GenAI-Powered-PDF-Knowledge-Assistant](Github)
cd GenAI-Powered-PDF-Knowledge-Assistant

### 2. Create and Activate Virtual Environment 
# Create environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

### 3. Install Dependencies
```Bash
pip install -r requirements.txt

### 4. Setup Environment Variables
Create a .env file in the root directory and paste your Gemini API key:
GOOGLE_API_KEY="your_actual_gemini_api_key_here"

### 5. Run the Application
```Bash
streamlit run app.py

# 🧠 How It Works (RAG Pipeline) 
1. Upload: User uploads a corporate policy, resume, or textbook PDF.
2. Chunking: The system breaks the long document into small semantic chunks of 1000 characters.
3. Embeddings: Text chunks are transformed into high-dimensional vectors and indexed in FAISS.
4. Retrieval: When a question is asked, FAISS performs a similarity search to fetch the top 4 most relevant chunks.
5. Generation: The context chunks along with the query are passed to the Gemini LLM to generate a pinpoint, clean response.

