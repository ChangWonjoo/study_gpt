import streamlit as st
import time
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    loader = UnstructuredFileLoader("./.cache/files/Genesis.txt")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        # length_function=len,
    )

    embeddings = OpenAIEmbeddings()

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings,cache_dir,)


    docs = loader.load_and_split(text_splitter=splitter)

    vector_store = FAISS.from_documents(docs, cached_embeddings)

    retriever = vector_store.as_retriever()
    return retriever

st.title("DocumentGPT")

st.markdown("""
Welcome!
Use this chatbot to ask questions about the DocumentGPT model.
""")

file = st.file_uploader("Upload a .txt .pdf or .docx file", 
                        type=["txt", "pdf", "docx"])

if file:
    # st.write("File uploaded")
    # st.write(file)
    retriever = embed_file(file)
    s = retriever.invoke("What is the document about?")
    st.write(s)