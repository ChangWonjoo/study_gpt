import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings

from langchain.vectorstores import Chroma, FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-3.5-turbo",
)

st.cache_data(show_spinner="Loading file...")
def spilt_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
        # length_function=len,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    return docs

with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        ("File","Wikipedia Aricle")
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a file", 
            type=["txt", "pdf", "docx"],
        )
        if file:
            docs = spilt_file(file)
    else:
        topic = st.text_input("Search Wikipedia Article")
        if topic:
            retriever = WikipediaRetriever()
            with st.status("Searching Wikipedia Article..."):
                docs = retriever.get_relevant_documents(topic)
                # st.write(docs)

if not docs:
    st.markdown(
        """
                    
        Welcome to QuizGPT.
                    
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                    
        Get started by uploading a file or searching on Wikipedia in the sidebar.
                    
    """
    )
else:
    st.write(docs)