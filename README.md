# 📚 Study Buddy – AI-Powered RAG Application

Study Buddy is an AI-powered Retrieval-Augmented Generation (RAG) application designed to answer questions using information retrieved from documents. It combines semantic search, vector databases, reranking, and locally running Large Language Models (LLMs) to generate context-aware and reliable answers.

The project implements an end-to-end RAG pipeline using **ChromaDB, Sentence Transformers, Cross-Encoder, and Llama 3.2 through Ollama**. It also integrates **RAGAS with NVIDIA NIM** to evaluate the quality of generated answers and retrieved context.

## 🚀 Features

- **Document Chunking:** Splits documents into smaller, meaningful chunks using LangChain's RecursiveCharacterTextSplitter.
- **Text Embeddings:** Generates vector embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
- **Vector Database:** Stores and retrieves document embeddings using persistent ChromaDB storage.
- **Semantic Retrieval:** Finds relevant document chunks using cosine similarity.
- **Cross-Encoder Reranking:** Reorders retrieved chunks to improve contextual relevance.
- **LLM Generation:** Uses Llama 3.2 through Ollama for locally generated answers.
- **Context-Grounded Responses:** Generates answers using retrieved context and instructs the model not to invent missing information.
- **Interactive Chat:** Provides a command-line interface for asking questions.
- **RAGAS Evaluation:** Evaluates RAG performance using NVIDIA NIM as the evaluation model.
- **CSV Reports:** Exports detailed evaluation results for further analysis.

## 🏗️ RAG Architecture

```text
             Input Documents
                    |
                    v
             Text Chunking
                    |
                    v
             Text Embeddings
                    |
                    v
               ChromaDB
                    |
                    |
User Question ------+
       |
       v
  Semantic Retrieval
       |
       v
 Cross-Encoder Reranking
       |
       v
  Relevant Context
       |
       v
   Prompt Building
       |
       v
  Llama 3.2 (Ollama)
       |
       v
  Context-Aware Answer
       |
       v
     User
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Ollama | Local LLM execution |
| Llama 3.2 | Answer generation |
| ChromaDB | Vector database |
| Sentence Transformers | Text embeddings |
| LangChain | Recursive text chunking |
| Cross-Encoder | Retrieval reranking |
| RAGAS | RAG evaluation |
| NVIDIA NIM | Evaluation LLM and embeddings |

## 📊 RAG Evaluation

The project integrates RAGAS to evaluate the retrieval and generation pipeline using four metrics:

| Metric | Description |
|---|---|
| Faithfulness | Measures whether answers are supported by retrieved context |
| Answer Relevancy | Measures how relevant the generated answer is to the question |
| Context Precision | Measures the relevance of retrieved context |
| Context Recall | Measures whether relevant information is retrieved |

The evaluation uses:
- **Judge LLM:** `meta/llama-3.1-70b-instruct`
- **Judge Embeddings:** `nvidia/nv-embedqa-e5-v5`
- **Evaluation Provider:** NVIDIA NIM

Evaluation results are displayed in the terminal and exported to `ragas_results.csv`.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Lekhana-dama/Study-Buddy-RAG.git
cd Study-Buddy-RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install ollama chromadb sentence-transformers langchain-text-splitters
```

For RAGAS evaluation:

```bash
pip install ragas datasets langchain langchain-nvidia-ai-endpoints
```

### 4. Install Ollama and download the model

Install Ollama from:

https://ollama.com/

Download Llama 3.2:

```bash
ollama pull llama3.2
```

Make sure the Ollama service is running.

### 5. Configure NVIDIA API Key

Set your NVIDIA API key as an environment variable if you want to run RAGAS evaluation.

PowerShell:

```powershell
$env:NVIDIA_API_KEY="your_api_key"
```

Get an API key from:

https://build.nvidia.com/

**Note:** Normal RAG question answering uses local Ollama and does not require an NVIDIA API key. The key is needed for NVIDIA-powered RAGAS evaluation.

## ▶️ Running the Application

Run the Python application:

```bash
python main.py
```

The application will:
1. Initialize the embedding model and reranker.
2. Create or load the persistent ChromaDB collection.
3. Ingest sample documents if the collection is empty.
4. Execute demonstration questions.
5. Start the interactive chat interface.

### Interactive Commands

| Command | Function |
|---|---|
| `info` | Displays the number of indexed chunks |
| `eval` | Runs RAGAS evaluation |
| `quit` | Exits the application |

## 📁 Project Structure

```text
Study-Buddy-RAG/
│
├── main.py
├── chroma_db_local/
├── .env
├── .gitignore
└── README.md
```

*The structure above represents the main application and its local runtime data. Adjust filenames if your repository uses a different layout.*

## 🔮 Future Improvements

- PDF and multiple document format ingestion.
- Web-based chat interface.
- Conversation history and memory.
- Support for multiple LLM models.
- Improved retrieval strategies.
- Expanded evaluation datasets and benchmarking.

## 🎯 Learning Outcomes

Through this project, I explored and implemented:

- End-to-end Retrieval-Augmented Generation.
- Semantic search and vector similarity.
- Embedding generation and vector storage.
- Retrieval optimization using Cross-Encoder reranking.
- Local LLM integration using Ollama.
- RAG evaluation using RAGAS and NVIDIA NIM.

## 👩‍💻 Author

**Lekhana Dama**

GitHub: https://github.com/Lekhana-dama

---

⭐ If you find this project interesting, consider giving the repository a star!
