import asyncio
import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer


asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.title("SiteGPT")

html2text_transformer = Html2TextTransformer()

st.markdown(
    """
    This app uses GPT-3 to generate text based on the content of a website.
    """
)

with st.sidebar:
    url = st.text_input(
        "Write down a Url",
        placeholder="https:/example.com",
    )

if url:
    #async chromium loader
    loader = AsyncChromiumLoader([url],headless=True)
    # docs = asyncio.run(loader.load())
    docs = loader.load()
    transformed_docs = html2text_transformer.transform_documents(docs)
    st.write(transformed_docs)