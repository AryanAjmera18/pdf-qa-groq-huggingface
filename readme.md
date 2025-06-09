# 🧠 PDF Q\&A Assistant with GROQ + HuggingFace Embeddings

This project is an intelligent question-answering system that allows users to query across a set of PDF documents using LLMs from **GROQ (LLaMA3)** and semantic embeddings from **HuggingFace (sentence-transformers)**. Built with **LangChain**, **FAISS**, and **Streamlit**, this app provides fast, context-aware answers with document similarity display.

🌐 **Live Demo:** [Streamlit App](https://aryanajmera18-pdf-qa-groq-huggingface-app-50pdvs.streamlit.app/)

---

## 🚀 Features

* 🔍 Uploads and parses multiple PDFs from the `research_papers/` folder
* 🧠 Embeds chunks using `all-MiniLM-L6-v2` from HuggingFace
* 🗃️ Creates a FAISS vector store for fast semantic retrieval
* 💬 LLM-powered answers using GROQ’s `LLaMA3-8b-8192`
* ⚡ Measures response time for each query
* 📂 View retrieved document chunks for transparency

---

## 📦 Tech Stack

* **Streamlit** – for the web interface
* **LangChain v0.2+** – to create retrieval-augmented generation chains
* **GROQ + LLaMA3** – for fast LLM inference
* **HuggingFace Embeddings** – for sentence-level semantic representation
* **FAISS** – for efficient vector search
* **dotenv** – for secure API key management

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/aryanajmera18/pdf-qa-groq-huggingface.git
cd pdf-qa-groq-huggingface
```

### 2. Create `.env` File

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure these libraries are installed:

* `langchain`
* `langchain-community`
* `langchain-groq`
* `langchain-huggingface`
* `streamlit`
* `faiss-cpu`
* `sentence-transformers`

### 4. Add Your PDFs

Place all PDF files in the `research_papers/` directory.

### 5. Run the App

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. Click **"Document Embedding"** to create the vector store from PDFs
2. Ask a natural-language question
3. The app:

   * Retrieves relevant chunks using vector similarity
   * Passes them with the question to GROQ’s LLaMA3
   * Displays the best answer and supporting document snippets

---

## 📁 Project Structure

```
pdf-qa-groq-huggingface/
├── app.py
├── research_papers/           # Place your PDFs here
├── requirements.txt
└── .env                       # Your API keys (not tracked by git)
```

---

## 📄 Example Use Cases

* Research assistants for academic papers
* Enterprise document Q\&A systems
* Legal or financial report querying

---

## 📝 License

MIT License

---

## 🤝 Contributions

PRs welcome! Please submit issues and enhancements via GitHub.

---

## 🔗 Acknowledgments

* [LangChain](https://github.com/langchain-ai/langchain)
* [GROQ](https://groq.com)
* [HuggingFace](https://huggingface.co)
* [FAISS](https://github.com/facebookresearch/faiss)

---

## 📛 Suggested GitHub Repo Name

```
pdf-qa-groq-huggingface
```
