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

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_data(show_spinner="Embedding the file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    # st.write(file_content, file_path)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    loader = UnstructuredFileLoader(file_path)

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

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})

def paint_history():
    # st.write(st.session_state)
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)



st.title("DocumentGPT")

st.markdown("""
Welcome!
Use this chatbot to ask questions about the DocumentGPT model.
            
            Please upload file on side bar a .txt, .pdf or .docx file to get started.
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", 
                            type=["txt", "pdf", "docx"])

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask a question!", "ai", save=False)
    paint_history()

    message = st.chat_input("ask anything about your file")
    if message:
        send_message(message, "human")

else:
    #채팅을 중단하기 위해 파일을 삭제하면, 대화창을 없애는 동시에 대화기록을 초기화 한다.
    st.session_state["messages"] = []  