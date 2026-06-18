# RAG---Internship
# End-to-End Retrieval-Augmented Generation Pipeline

## Overview

This project implements a complete Retrieval-Augmented Generation (RAG) system using:

- Google Gemini Embeddings
- ChromaDB Vector Database
- LangChain
- LangGraph
- Groq LLMs (Llama 3.3 70B)
- RAGAS Evaluation Framework

The system ingests multiple PDF documents, converts them into embeddings, stores them in a persistent vector database, retrieves relevant context for user queries, and generates grounded responses using a Large Language Model.

---

## Features

### Document Processing
- Loads multiple PDF documents
- Extracts and preprocesses text
- Handles file loading errors safely

### Chunking Strategy
- Recursive text splitting
- Optimized chunk sizes
- Overlapping chunks for context preservation

### Vector Database
- Google Gemini Embedding Model
- ChromaDB persistent storage
- Efficient similarity search

### Retrieval Pipeline
- Semantic document retrieval
- Top-k relevant chunk selection
- Context preparation for generation

### Generation Pipeline
- Groq-hosted Llama 3.3 70B model
- Prompt-based answer generation
- Context-grounded responses

### Workflow Orchestration
- LangGraph state-based workflow
- Modular retrieval and generation nodes
- Scalable architecture

### Evaluation
- RAGAS-based evaluation
- Faithfulness measurement
- Answer Relevancy analysis
- Context Precision scoring
- Context Recall scoring

---

## Tech Stack

| Component | Technology |
|------------|------------|
| Framework | LangChain |
| Workflow | LangGraph |
| Embeddings | Gemini Embedding 001 |
| Vector Store | ChromaDB |
| LLM | Groq Llama 3.3 70B Versatile |
| Evaluation | RAGAS |
| Language | Python |

---

## Project Architecture

```text
PDF Documents
      │
      ▼
Document Loader
      │
      ▼
Text Chunking
      │
      ▼
Gemini Embeddings
      │
      ▼
ChromaDB Vector Store
      │
      ▼
Retriever
      │
      ▼
LangGraph Workflow
      │
 ┌────┴────┐
 ▼         ▼
Retrieve  Generate
 Node      Node
      │
      ▼
Final Answer
```

---

## Dataset

The system was built using PDF documents covering Artificial Intelligence topics such as:

- AI Applications in Healthcare
- AI Ethics and Bias
- Computer Vision
- Generative AI and Large Language Models
- Additional AI-related educational resources

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/rag-phase2.git

cd rag-phase2
```

Install dependencies:

```bash
pip install langchain
pip install chromadb
pip install pypdf
pip install langchain-google-genai
pip install langchain-groq
pip install langgraph
pip install langchain-community
pip install langchain-text-splitters
pip install google-cloud-aiplatform
pip install langchain-google-vertexai
pip install ragas
pip install datasets
pip install pandas
```

Or:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Set the following API keys:

```bash
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

For Google Colab:

```python
from google.colab import userdata

gem_key = userdata.get("GEMINI_KEY")
groq_key = userdata.get("GROQ_KEY")
```

---

## Workflow

### Step 1: Load Documents

PDF files are loaded using:

```python
PyPDFLoader
```

---

### Step 2: Split Documents

Documents are divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

Benefits:

- Reduced token consumption
- Better retrieval quality
- Improved embedding efficiency

---

### Step 3: Generate Embeddings

Embeddings are generated using:

```python
GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
```

---

### Step 4: Store in ChromaDB

Embeddings are persisted locally:

```python
Chroma(
    persist_directory="./chroma_db"
)
```

---

### Step 5: Retrieve Relevant Context

Retriever configuration:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
```

The top 3 semantically relevant chunks are retrieved for every query.

---

### Step 6: Generate Response

The retrieved context is combined with the user query and passed to:

```python
ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)
```

The model generates a context-aware answer grounded in the retrieved documents.

---

## LangGraph Workflow

### Graph State

```python
class GraphState(TypedDict):
    question: str
    context: List[str]
    generation: str
```

### Nodes

#### Retrieve Node

Responsibilities:

- Accept user question
- Query ChromaDB
- Retrieve relevant chunks
- Update graph state

#### Generate Node

Responsibilities:

- Receive context
- Build prompt
- Generate final answer
- Return response

---

## Example Query

Input:

```text
What is Computer Vision?
```

Pipeline:

```text
Question
    ↓
Retriever
    ↓
Relevant Chunks
    ↓
Groq LLM
    ↓
Final Answer
```

Output:

```text
Computer Vision is a field of Artificial Intelligence that enables computers to interpret, analyze, and understand visual information from images and videos.
```

---

## Evaluation Using RAGAS

The project evaluates RAG quality using RAGAS metrics.

### Metrics Used

#### Faithfulness

Measures whether the generated answer is supported by retrieved context.

#### Answer Relevancy

Measures how relevant the answer is to the question.

#### Context Precision

Measures how much retrieved context is useful.

#### Context Recall

Measures whether sufficient relevant information was retrieved.

---

### Evaluation Pipeline

1. Prepare test questions
2. Generate answers through LangGraph
3. Collect retrieved contexts
4. Compare against ground-truth answers
5. Compute RAGAS metrics

Example:

```python
evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
    ]
)
```

---

## Results

The evaluation produces a scorecard containing:

| Metric | Description |
|----------|-------------|
| Faithfulness | Context grounding quality |
| Answer Relevancy | Query-answer alignment |
| Context Precision | Retrieval quality |
| Context Recall | Retrieval completeness |

These metrics help quantify both retrieval and generation performance.

---

## Folder Structure

```text
project/
│
├── PDFs/
│   ├── AI Applications in Healthcare.pdf
│   ├── AI Ethics and Bias.pdf
│   ├── Computer Vision.pdf
│   └── ...
│
├── chroma_db/
│
├── RAG_PHASE_2_FINAL.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## Future Improvements

- Hybrid Retrieval (BM25 + Dense Retrieval)
- Query Expansion
- Re-ranking Models
- Multi-query Retrieval
- Citation Generation
- Hallucination Detection
- Agentic RAG Architecture
- Real-time Web Retrieval Integration
- Conversational Memory Support

---

## Learning Outcomes

Through this project:

- Built a complete RAG pipeline from scratch
- Implemented semantic search using vector databases
- Integrated Gemini embeddings with ChromaDB
- Used Groq-hosted LLMs for generation
- Designed workflows using LangGraph
- Evaluated system quality using RAGAS
- Developed an end-to-end production-style GenAI application

---

## Authors

RAG Phase 2 Project

Built using LangChain, LangGraph, ChromaDB, Gemini Embeddings, Groq LLMs, and RAGAS Evaluation Framework.

---
