# AmbedkarGPT Q&A System

A simple command-line RAG system that answers questions based on Dr. B.R. Ambedkar's speech using LangChain, ChromaDB, and Ollama.

## Prerequisites

- Python 3.8 or higher
- Ollama installed with Mistral model

## Setup Instructions

### 1. Install Ollama and Mistral Model

Download and install Ollama from https://ollama.com/download

Then pull the Mistral model on terminal:

```
ollama pull mistral
```

Verify it's running:
```
ollama run mistral
```

Type `/bye` to exit the Ollama chat interface.

### 2. Clone or Create Project Directory

```
mkdir AmbedkarGPT-Intern-Task
cd AmbedkarGPT-Intern-Task
```

### 3. Create Virtual Environment

```
python -m venv .venv
```

Activate the virtual environment:

Windows (PowerShell):
```
.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):
```
.venv\Scripts\activate.bat
```

Linux/Mac:
```
source .venv/bin/activate
```

### 4. Install Dependencies
```
pip install -r requirements.txt
```

## Usage

### First Run

On the first run, the system will create embeddings and store them in ChromaDB:
This will:
- Load the speech.txt file
- Split it into chunks
- Create embeddings using sentence-transformers
- Store them in a local ChromaDB vector store
- Start the Q&A interface

### Subsequent Runs

The system will load the existing vector store, making startup faster:

python main.py
### Example Questions

- Question: What is the real remedy according to the text?
- Question: What does the text say about the shastras?
- Question: How does the text describe social reform work?

Type `exit` or `quit` to stop the program.

## Project Structure
```
AmbedkarGPT-Intern-Task/
├── main.py # Main application code
├── requirements.txt # Python dependencies
├── README.md # This file
├── speech.txt # Input text file
├── chroma_db/ # ChromaDB vector store (created automatically)
└── .venv/ # Virtual environment (created by us)
```


## How It Works

1. **Document Loading**: Loads speech.txt using TextLoader
2. **Text Splitting**: Splits text into 300-character chunks with 50-character overlap
3. **Embeddings**: Creates vector embeddings using sentence-transformers/all-MiniLM-L6-v2
4. **Vector Store**: Stores embeddings in ChromaDB for efficient retrieval
5. **Retrieval**: Finds the 3 most relevant chunks for each question
6. **Generation**: Feeds context and question to Ollama Mistral to generate answers

## Technologies Used

- **LangChain**: Orchestration framework for RAG pipeline
- **ChromaDB**: Local vector database for embeddings
- **HuggingFace Embeddings**: sentence-transformers/all-MiniLM-L6-v2 model
- **Ollama**: Local LLM runtime with Mistral 7B model

## Note:

Make sure to run the ollama in the background