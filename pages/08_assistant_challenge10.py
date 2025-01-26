# challenge10(final): OpenAI 어시스턴트로 리팩터링
# 이전 과제에서 만든 에이전트를 OpenAI 어시스턴트로 리팩터링합니다.
    # CHALLENGE 9 내용
    # Wikipedia / DuckDuckGo에서 검색
    # 웹사이트의 텍스트를 스크랩하고 추출합니다.
    # 리서치 결과를 .txt 파일에 저장하기
    # 다음 쿼리로 에이전트를 실행합니다: "Research about the XZ backdoor" 라는 쿼리로 에이전트를 실행하면, 
    # 에이전트는 Wikipedia 또는 DuckDuckGo에서 검색을 시도하고,
    #  DuckDuckGo에서 웹사이트를 찾으면 해당 웹사이트에 들어가서 콘텐츠를 추출한 다음 
    # .txt 파일에 조사 내용을 저장하는 것으로 완료해야 합니다.
# 대화 기록을 표시하는 Streamlit 을 사용하여 유저 인터페이스를 제공하세요.
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
# st.sidebar를 사용하여 Streamlit app 의 코드과 함께 깃허브 리포지토리에 링크를 넣습니다.

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler

st.set_page_config(
    page_title="Assistant Challenge 10",
    page_icon="🖥️",
)

st.markdown(
    """
    # Assistant GPT
    
    CHALLENGE 9 내용
    1. Wikipedia / DuckDuckGo에서 검색
    2. 웹사이트의 텍스트를 스크랩하고 추출합니다.
    3. 리서치 결과를 .txt 파일에 저장하기
    4. 다음 쿼리로 에이전트를 실행합니다: "Research about the XZ backdoor" 라는 쿼리로 에이전트를 실행하면, 
    5. 에이전트는 Wikipedia 또는 DuckDuckGo에서 검색을 시도하고, 
        DuckDuckGo에서 웹사이트를 찾으면 해당 웹사이트에 들어가서 콘텐츠를 추출한 다음 
        .txt 파일에 조사 내용을 저장하는 것으로 완료해야 합니다.
"""
)


with st.sidebar:
    st.markdown("[Link to the code on GitHub](https://github.com/ChangWonjoo/study_gpt/commit/ea7993941af0e414bba8d720ad18c407e51c9968#diff-ccdccc66ba501ee6d447119afd718517de132cc7d9469bf36272ce0e13b9c120)")

    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if openai_api_key:
        st.session_state["openai_api_key"] = openai_api_key
    # st.write(st.session_state)

    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

# OpenAI API 키 설정
if "openai_api_key" in st.session_state:
    llm = ChatOpenAI(
        openai_api_key=st.session_state["openai_api_key"],
        temperature=0.1,
        # model_name="gpt-3.5-turbo",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

else:
    st.error("Please enter your OpenAI API Key in the sidebar.")



