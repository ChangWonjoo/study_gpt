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
from openai import OpenAI
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import yfinance
import json

##################################
# tool function definition
##################################
def get_ticker(inputs):
    company_name = inputs["company_name"] #inputs에는 {"company_name": "Apple"} 이런식으로 들어옴
    # ddg = DuckDuckGoSearchAPIWrapper()
    # return ddg.run(f"Ticker symbol of {company_name}")
    ticker = "AAPL"
    return {"ticker": ticker}

def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    # return stock.income_stmt #pandas dataframe
    return json.dumps(stock.income_stmt.to_json()) #json string

def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())

def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json()) #json string

functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": "Given the name of a company returns its ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's income statement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's balance sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_performance",
            "description": "Given a ticker symbol (i.e AAPL) returns the performance of the stock for the last 100 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]

##################################
# create assistant & save assistant_id
##################################
# assistant = client.beta.assistants.create(
#     name="Investor Assistant",
#     instructions="You help users do research on publicly traded companies and you help users decide if they should buy the stock or not.",
#     model="gpt-4-1106-preview",
#     tools=functions,
# )
assistant_id='asst_gkcE5OgmIiWFufkYVqy1CVRk'


##################################
# streamit page
##################################
# 페이지 구성
st.set_page_config(
    page_title="Assistant Challenge 10",
    page_icon="🖥️",
)
st.markdown(
    """
    # Assistant GPT (challenge 10)
    
    [ ] 이전 과제에서 만든 에이전트를 OpenAI 어시스턴트로 리팩터링합니다.
        
    CHALLENGE 9 내용
    >> 

        1. Wikipedia / DuckDuckGo에서 검색
        2. 웹사이트의 텍스트를 스크랩하고 추출합니다.
        3. 리서치 결과를 .txt 파일에 저장하기
        4. 다음 쿼리로 에이전트를 실행합니다: "Research about the XZ backdoor" 라는 쿼리로 에이전트를 실행하면, 
        5. 에이전트는 Wikipedia 또는 DuckDuckGo에서 검색을 시도하고, 
            DuckDuckGo에서 웹사이트를 찾으면 해당 웹사이트에 들어가서 콘텐츠를 추출한 다음 
            .txt 파일에 조사 내용을 저장하는 것으로 완료해야 합니다.
    [ ] 대화 기록을 표시하는 Streamlit 을 사용하여 유저 인터페이스를 제공하세요.

    [o] 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.

    [ ] st.sidebar를 사용하여 Streamlit app 의 코드과 함께 깃허브 리포지토리에 링크를 넣습니다.
"""
)

# OpenAI API 키 설정
if "openai_api_key" in st.session_state:
    client = OpenAI()
else:
    st.error("Please enter your OpenAI API Key in the sidebar.")

    
#사이드바 작성
with st.sidebar:
    st.markdown("[Link to the code on GitHub](https://github.com/ChangWonjoo/study_gpt/commit/ea7993941af0e414bba8d720ad18c407e51c9968#diff-ccdccc66ba501ee6d447119afd718517de132cc7d9469bf36272ce0e13b9c120)")

    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if openai_api_key:
        st.session_state["openai_api_key"] = openai_api_key
        st.write(st.session_state)

    url = st.text_input(
        "Write down a Company name to research",
        placeholder="Apple Inc.",
    )

st.write("========================================")
st.write("url input:", url)


# thread 생성
if "thread_id" in st.session_state:
    # thread = client.beta.threads.retrieve(st.session_state["thread_id"])
    thread = st.session_state["thread"]
else:
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "I want to know if the Apple stock is a good buy",
            }
        ]
    )
    st.session_state["thread"] = thread
st.write(st.session_state['thread'])

# st.write("========================================")
# st.write("thread_id:", thread.id)
# st.write("run_id:", st.session_state["run"].id)
# run_retrieve = client.beta.threads.runs.retrieve(run_id = st.session_state["run"].id, thread_id=thread.id)
# st.write("run_retrieve:", run_retrieve)
if "run_id" in st.session_state:
# if run_retrieve.status is "in_progress":
    run = st.session_state["run"]
else:
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    st.session_state["run"] = run

st.write(st.session_state['run'])


##################################
# assistant function definition
##################################
def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )

def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages) #list 매소드를 사용하기 위해 객체를 list로 변환
    messages.reverse()
    for message in messages:
        print(f"{message.role}: {message.content[0].text.value}")

def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread.id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        # because function.arguments just brings str, so convert it to json so that the function can actually use it.
        function_args = json.loads(function.arguments)
        output = functions_map[function.name](function_args)
        output_str =json.dumps(output)
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": output_str,
                "tool_call_id": action_id,
            }
        )
    return outputs

def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outputs
    )


get_run_result = get_run(run.id, thread.id).status
st.write("get_run_result:", get_run_result)

messages = get_messages(thread.id)
st.write("messages:")
st.write(messages)

submit_tool = submit_tool_outputs(run.id, thread.id)
st.write("submit_tool: ")
st.write(submit_tool)