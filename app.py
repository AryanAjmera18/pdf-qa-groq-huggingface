import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Load LLM
llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model="Llama3-8b-8192")

# Prompt must have both 'input' and 'context'
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question based on the given context."),
    ("human", "Context:\n{context}\n\nQuestion:\n{input}")
])

# Embedding + FAISS setup
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )
        st.session_state.vector_store = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
        st.success("✅ Vector database created successfully!")

# Streamlit UI
st.title("📄🔍 PDF Q&A with GROQ + HuggingFace")
user_prompt = st.text_input("💬 Enter your question")

if st.button("📚 Create Document Embedding"):
    create_vector_embedding()

if user_prompt and "vector_store" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    end = time.process_time()

    st.write("🧠 **Answer:**")
    st.success(response["answer"])
    st.write(f"⏱️ Response time: `{end - start:.2f} seconds`")

    with st.expander("📄 Retrieved Document Chunks"):
        for i, doc in enumerate(response.get("context", [])):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content)
            st.markdown("---")
else:
    if user_prompt:
        st.warning("⚠️ Please embed documents first by clicking the button above.")
