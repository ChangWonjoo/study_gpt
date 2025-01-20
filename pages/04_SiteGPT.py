import asyncio
import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import SitemapLoader
import logging
from fake_useragent import UserAgent

asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# Initialize a UserAgent object
ua = UserAgent()

@st.cache_data(show_spinner="loading website...")
def load_website(url):
    loader = SitemapLoader(url)
    loader.requests_per_second = 1
    # Set a realistic user agent
    loader.headers = {'User-Agent': ua.random}
    docs = loader.load()
    return docs

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
    # #async chromium loader##############
    # loader = AsyncChromiumLoader([url],headless=True)
    # docs = loader.load()
    # transformed_docs = html2text_transformer.transform_documents(docs)
    # st.write(transformed_docs)
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        docs = load_website(url)
        st.write(docs)