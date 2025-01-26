# study_gpt
for studying LLM AI with nomad coder..


# 241215-1
    1. codespace 설치 후 확인해보니, python 3.12로 설치 되어있어, 설정 바꾸려니 복잡해보여서 로컬로 먼저 해보기로 함.

    2. 로컬 PC도 3.12로 되어있어서 3.11.9버젼 설치 진행
    https://www.python.org/downloads/release/python-3119/

    3. python3.11 -m venv ./env && python3.11 --version 코멘드 찾을 수 없다는 에러 발생 (패스 설정 문제일듯?)
    py -3.11 -m venv ./env 로 env생성 완료.
    >> ./env/Scripts/activate로 가상환경 실행 (나갈 때는 deactivate)

    4. pip install -r requirements.txt >> 설치 확인용 main.py로 ticktoken 라이브러리 위치 확인되면 완료(<module 'tiktoken' from 'C:\\Users\\wonjooLAPTOP\\Dropbox\\github\\study_gpt\\env\\Lib\\site-packages\\tiktoken\\__init__.py'>).

    5. .env 파일 생성 후 설정 값 작성.

# 241215-2
    #3.0강의
    1.jupyter Notebooks.
    2.openaiplatform에 $5결제 후, APIkey 등록
    3.llm / chatmodel 테스트 진행.

# 241217
    #3.1 Predict Messages
    1. 질문 내용 하나만을 predict 하는 것이 아닌, 문맥을 읽을 수 있도록 시스템설정, AI가 이야기한 내용, 사람이 이야기한 내용을 모아서 전달하고 문맥에 맞게 답을 내놓도록 하는 것.
    - SytemMessage(시스템설정) / AIMessage(AI가 한 말) /HumanMassage(사람이 한 말)

# 241220
    #3.2 Prompt Template
    1. Prompt Template(String = 설정활용) / ChatPromptTempalte(Messages = 문맥활용)가 있고, 각각 장단점이 있다.
    - PromptTemplate - 템플릿을 포맷해서 나온 프롬프트를 ChatAI에게 "String"으로 전달하여 Predict
        prompt = template.format(country_a = "korea", country_b = "japan")
    - ChatPromptTemplate - 템플릿을 포맷해서 나온 프롬프트를 ChatAI에게 "Messages"로 전달하여 Predict
        prompt = template.format_messages(launguage="Korean", name="고구", country_a="Seoul", country_b="Busan")

# 241221
    #3.3 OutputParser and LCEL(lang chain expression launguage)
    1. Parser가 필요한 이유는 LLM의 응답을 저장하는 등 변경해서 사용해야 할 떄가 있기 떄문이다.
        #strip() - remove whitespace, split() - split by comma
    2. Chain 이라는 것은 기본적으로, 모든 요소를 합쳐주는 역할.
        ex. chain = template | chat | CommaOutputParser()
        template 지정 >> formating 해서 chatAI에게 전달 >> parser 사용해서 결과 얻기의 이련의 과정을 묶어서 표현.
    
    #3.4 Chaining Chains >> notebook1.ipynb에서 진행
        chef_chain = chef_prompt | chat
        veg_chain = veg_chef_pprompt | chat
        >> final_chain = {"recipe":chef_chain} | veg_chain　
           ＃다음 요소에 결과값을 전달하는 구조 : RunableMap이라고 하는 랭체인언어.
        >> final_chain.invoke({"cuisine":"italian",})

    #3.5 Recap

# 241222
    #4.0 Introduction

# 241224
    #4.1 FewShotPromptTemplate
    fewshot은 모델에게 더 좋은 대답을 할 수 있도록 참고할 수 있는 좋은 대답 예제를 준다는 뜻.(systemMessage로 설정하던 것을 예제를 통해 설정하는 것으로 변경할 수 있다.)

    #{question} 과 {answer}의 변수 이름을 예저와 잘 맞춰줘야 한다.
    예제의 형식을 지정해주는 방법>>
        example_template = """
            Human:{question}
            AI:{answer}
        """
        prompt = FewShotPromptTemplate(
            example_template=example_template,
            examples=examples,
            suffix="Hhuman: What do you know about {country}?",
            input_variable=["country"],
        )

# 241228
    #4.2 FewShotChatMessagePromptTemplate
    examples를 이용해 시스템 메세지, *형식화된 예제*, 질문이 포함된 나만의 최종 프롬프트를 만든다.
    >> chatpromptTemplate.from_message안에 examples와 example_template로 만든 example_prompt를 넣어서 프롬프트를 포맷팅

# 250101
    #4.3 LengthBaseExampleSelector
    LengthBaseExampleSelector : 사용자의 종류, 로그인 여부, 사용하는 모델에 따라 활용하는 예제의 양을 조절할 수 있다.
    RandomExapleSelector : 저장해둔 예제 중 하나를 랜덤으로 사용하도록 가져올 수 있다.

    #4.4 Serialization and Composition
    load_prompt : 외부 Prompt를 활용하는 방법 (josn / yaml)
    PipelinePromptTemplate : 많은 프롬프트들의 memory를 모으는 방법
    * #연결 해주는 프롬프트의 형태를 설명하고, 리스트 형태로 조각 프롬프트를 넣어서 완성된 프롬프트를 만든다.

# 250106
    -- notebook4 -- 
    #4.5 Chaching
    같은 질문이 반복적으로 올 확률이 높다면, 캐시를 이용해서 정해진 답을 반복 제출하게 함으로써 효율을 높일 수 있다. >> set_llm_cache(InMemoryCache()) 
    개발과정에서 챗모델의 작업 과정이나 결과물 도출을 위해 사용된 설정 및 수치를 확인 할 수 있다. >> set_debug(True)

    #4.6 Serialization
    챗모델을 사용하는 비용을 추정할 수 있다. >> with get_openai_callback() as usage:
    llm모델의 설정을 저장하고 불러올 수 있다. >> 저장: chat.save("models.json") //

# 250107 - 아직 강의 실습 안함
    #5.1~5.4 ConversationMemory
    대화 모델은 이전 내용을 기억하고 이어나가야지 사람과 대화하는 느낌이 든다.
    chat모델은 메모리를 지원하지 않기 때문에 랭체인 메모리 기능을 활용해야한다.
    랭체인에서 지원하는 메모리 모델은 총 5가지가 있다(API는 모두 동일).
        1. ConversationBufferMemory - 단순히 대화내용을 저장 ( 대화가 길어질 수록 메모리가 커져서 비효율적)
        2. ConversationBufferWindowMemory - 한도 수량 내에서 최근 대화 내용을 저장 ( 오래된 것 부터 순차적으로 삭제하여, 메모리 크기 유지) 
        3. ConversationSummaryMemory - llm사용(메모리 사용에 비용필요), chat모델을 이용해서 대화내용을 요약하여 저장. 대화가 길어질 수록 유용하다.
        4. ConversationSummaryBufferMemory - 일정 개수가 넘어가면 요약(systemMessage)을 시작 
        5. ConversationKnowkedgeGraphMemory - 대화이력에서 중요한 내용(entity)만 요약하여 저장

# 250108 - 아직 강의 실습 안함
    #5.8 recap
    <프롬프트에 메모리를 추가하는 3가지 방법>
    LLM chain >>
    Chat prompt template >>
    수동 메모리 관리법 (권장) >>  
    * 메모리를 저장한 뒤에 메모리로 활용할 수 있는 것들이 있기 때문

# 250109
    #5.5 Memory on LLMChain
    LLMChain (Off-the-shelf chain) 일반적인 목적을 가진 chain을 의미.
    chain = LLMChain(
        llm=llm,memory=memory,prompt= PromptTemplate.from_template("{question}")
        verbose=True, # chain프롬프트 로그 확인
    )
    프롬프트 템플릿에 대화의 기록을 넣을 자리를 만들어주고, 그 자리에 memoryClass의 memory_key를 이용해서 넣어준다. >> memory = ConversationSummaryBufferMemory(llm=llm,max_token_limit=120 
    memory_key="chat_history",)

    #5.6 ChatBasedMemory 
    대화형 메세지는 문자열방식/메세지방식 중 고를 수 있다.
     >> return_messages=True, 속성을 통해 문자열로 받을지 메세지로 받을지 선택 가능
    얼마나 많은 대화내용이 들어올 지 모르기 때문에 공간을 만든다. 
     >> MessagePlaceholder(variable_name = memory에서 사용한 변수이름)

    #5.7 LCEL BasedMemory
    load_memory_bariables({})['history'] >> 저장해둔 대화 혹은 요약(메모리)을 로드 시킬 수 있다. 
    save_context >> HumanMessages(Input)와 AIMessages(Output)를 메모리에 저장한다.
    RunnablePassthrough.assign()함수를 LCEL 맨 앞에 넣음으로써, 프롬프트 포맷팅 전에 필요한 함수를 실행시키고, 해당 함수에 필요한 값을 변수에 할당 할 수 있게 된다.


# 250112
    #6.0 Introduction
        RAG(검색 증강 생성) 모델이 일반적으로 접근할 수 없는 우리의 개인 정보 혹은 개인 자료를 컨텍스트로써 질문과 함께 제공하여 랭귀지 모델을 확장시켜 검색하는 방법
        
    #6.1 DataLoaders and Spitters
        RAG - 랭체인 모듈 Retrival ( Source >> Load Data >> Transform(split) >> Embed >> Store >> Retrieve  )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, #자르는 크기 기준
            chunk_overlap=50, # 자를 때 중복되는 크기
        )
    #6.2 Tiktoken
        token:모델이 단어를 해석하는 단위.
        우리는 단어의갯수, 문자의 갯수등으로 정보량을 세지만, llm은 토큰의 개수로 정보량을 센다.
        ***질문***
        Q.모델은 token단위로 단어를 읽고, 사람은 단어 단위로 읽는다고 하셨는데요.
        3:49에서 쌤이 말씀하시는 "우리가 세는 방법과 모델이 세는 방법이 같다"는 표현은 어떤 것을 이야기하는 것일까요?
        A.The way we count words in a sentence is roughly the way the model counts tokens in a sentence, not 100% but an approximation.

    #6.3 Vectors
        https://turbomaze.github.io/word2vecjson/
        https://www.youtube.com/watch?v=2eWuYf-aZE4

    #6.4 Vector Store
        vectorStore : 문서를 임배딩하여 벡터화하고, 벡터스토어를 활용해서 비슷한 문서를 찾을 수 있게 된다. 
        CacheBackedEmbeddings /LocalFileStore : 임배딩한 내용은 저장해서 사용하면 비용을 절약할 수 있다.

    #6.7 Recap
        LLM은 추가 자료를 작게 잘라줄 수록 비용, 속도, 검색 성능면에서 뛰어나지므로 작게 잘라서 줘야한다.
        Document를 적재(load)하고 분할(split)한다.
        Embedding - text에 의미별로 적절한 점수를 부여해서 vector형식으로 표현한 것. 
         >> OpenAIEmbeddings model사용하여 임베딩 된 내용을 Cache(저장)하였음

        stuff / map rerank, map reduce, refine

        vectorstore의 작동:
        MapReRank의 작동 : 도큐멘트를 순회하면서 각 도큐멘트에 기반해서 질문에 대답하고, 답변에 점수를 매겨줘

    #6.10 Recap

# 250113
    #6.5 RetrievalQA
    랭스미스 연동 안될 떄 **
        pip install -U langchain
        Pip install -U langchain-community

    #6.8 Stuff LCEL Chain
        RunnablePassthrough : input값을 그 다음 체인에게 전달해야할 때 사용한다. 

    #6.9 Map Reduce LCEL Chain
    RunnableLambda : chain과 그 내부 어디에서든 funciton을 호출할 수 있도록 해줌

    stuff 방식 : 질문을 query하여 관련있는 지문을 먼저 찾아서 찾은 지문 전체를 prompt로 넣고 llm을 돌린다.
    map reduce 방식 : 질문을 기반으로 retriever를 통해 찾은 질문과 연관된 지문 단락을 단락 수 만큼 for 문 수행하면서 각 단락 기준으로 llm 의 결과를 저장함. 각각 저장된 llm의 결과를 prompt로 합쳐서 llm 이 최종 결과를 만들어 냄.

# 250114
    #7.0 Introduction
    
    #7.1 Magic
    st.write() : write 함수를 사용해서 화면에 출력할 수 있다(형식은 알아서 맞춰준다).
    magic : st.wirte함수를 쓰지 않고 쥬피터 노트북처럼 변수명을 써놓는 것 만으로 동일하게 출력된다.
    
    #7.2 DataFlow
    data flow : 페이지 속 데이터 변경이 발생하면 python파일 전체가 재실행된다.

    #7.3 Multi Page
    streamlit은 자동적으로 pages폴더를 찾아서 참조한다.

    #7.4 Chat Messages
    session state는 여러번의 재실행에도 data가 보존될 수 있도록 해준다.

    #7.5 Recap

    #7.6 Uploading Documents
    @st.cache_data(show_spinner="Embedding the file...")
    함수 위에 데코레이션을 붙여서 file이 변경되지 않는 이상은 함수를 다시 실행하지 않고 저장된 return값을 바로 반환한다.
    이것은 streamlit이 기본적으로 파일 혹은 input값을 hash화 하여 가지고 있기 떄문이다.

    #7.7 Chat History 
    if file:
        retriever = embed_file(file)
        ..중략..
    else:
        #채팅을 중단하기 위해 파일을 삭제하면, 대화창을 없애는 동시에 대화기록을 초기화 한다.
        st.session_state["messages"] = []  

    #7.9 Streaming
    ChatCallbackHandler를 커스텀(app.py 내용 참조) : 답변이 실시간으로 작성되는 것처럼 보이는 방법
        class ChatCallbackHandler(BaseCallbackHandler):
        message = ""

        def on_llm_start(self, *args, **kwargs):
            self.message_box = st.empty()

        def on_llm_end(self, *args, **kwargs):
            save_message(self.message, "ai")

        def on_llm_new_token(self, token, *args, **kwargs):
            self.message += token
            self.message_box.markdown(self.message)


# 250117
    #9.0 Introduction - QuizGPT
    #9.2 GPT-4-turbo
    #9.3 question_chain
    #9.4 formatting_chain
        ```json{{ json내용 }}``` {}를 두개 겹쳐서 쓰는이유는 기본적으로 langchain에서 변수를 {변수명}으로 쓰기 때문에 ai가 변수명처럼 오해하지 않길 바래서이다.

    #9.5 output parser
    ** invoke()의 인자로 넣을 때는 바로로
        formatting_chain = formatting_prompt | llm
        formatting_response = formatting_chain.invoke(question_response.content)

    ** LCEL의 체인 안에 넣을 때는 .content속성을 찾아가기 때문에 붙이지 않는다.
        chain = {"context": question_chain} | formatting_chain | output_parser
    
    ** ```json ///내용/// ``` 이렇게 예시를 적어주면 AI가 쓸데없는 말을 앞뒤에 붙이지 않고 결과만을 반환한다.

    UnhashableParamError: Cannot hash argument 'docs' (of type builtins.list) in 'run_quiz_chain'.
    >> 해시할 수 없는 매개변가 있거나 stramilit이 데이터의 서명을 만들 수 없는 상황에서 함수의 인자를 cache를 만들기 위해 hasing을 시도하면 오류가 발생
    >> 함수 인자에 (_docs, topic) 방식으로 "_"값을 인자앞에 붙이면 해싱을 하지 않아 에러가 해결되나, 인자의 값이 바뀌어도 무시하게 도므로 추가인자를 통해 값이 바뀔 경우에는 해싱되도록 우회하여 작성한다.

    #9.6 
    리스트 컴프리헨션과 제너레이터 표현식 요약
    ** 리스트 컴프리헨션 : 리스트 컴프리헨션은 대괄호 []를 사용하여 리스트를 생성. 주어진 조건이나 변환을 적용하여 새로운 리스트를 생성하는 데 사용됩니다.
        리스트 컴프리헨션: [document["page_content"] for document in docs]는 docs 리스트의 각 document 객체에서 page_content 속성을 추출하여 새로운 리스트를 생성합니다.
        결과 출력: 변환된 리스트를 출력합니다.
    ** 제너레이터 표현식 : 제너레이터 표현식은 소괄호 ()를 사용하여 제너레이터 객체를 생성. 메모리 효율성을 높이기 위해 사용되며, 필요할 때마다 값을 하나씩 생성합니다.
        제너레이터 표현식: document["page_content"] for document in docs는 제너레이터 표현식입니다. 이는 docs 리스트의 각 document 객체에서 page_content 속성을 추출합니다.
        join 메서드: "\n\n".join(...)는 제너레이터 표현식에서 생성된 문자열을 "\n\n"로 결합합니다.
        결과 출력: 결합된 문자열을 출력합니다.

# 250120
    #10.1 AsyncChromiumLoader (가상환경활성화 >> playwright install 설치진행)
       AsyncChromiumLoader로 HTML가져오기 >> Html2TextTransformer로  HTML을 text로 변환
    ** chromium / playwright는 browser control(브라우저 컨트롤)을 할 수 있는 package.(selenium과 비슷) >> 대상 웹 사이트에 sitemap이 없거나 많은 양의 javascript를 가져서 javascript가 로드될 때까지 기다려야하는 웹 사이트(동적 웹사이트)로부터 자료를 가져올 떄 좋음.
    ** sitemap을 활용하는 방법 : 정적인 텍스트로 구성된 웹사이트에 좋음.

    NotImplementedError 발생 >> windows에서 실행하는 경우, 결과를 반환하기 위해 같이 돌아가야하는 함수가 제대로 동작하지 않아서 발생할 수 있다고 한다(?).
    해결책 : 아래 두 줄을 추가한다.
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

 # 250121
    https://www.google.com/forms/sitemaps.xml
    #10.2 SitemapLoader
    Empty 반환 에러 : pip install fake_useragent
    해결책>>
    from fake_useragent import UserAgent
    ua = UserAgent() # Initialize a UserAgent object
    loader = SitemapLoader(url)
    loader.headers = {'User-Agent': ua.random}

    #10.3 Parsing Function
    데이터를 가져올 URL을 필터링 하는 두 가지 방법
        1. 직접 URL 설정
        2. URL 필터링을 위한 정규식 설정
    헤더/풋터 제거 방법
        parsing_function속성 이용


# 250122
    #12.1 Your First Agent
    Agent는 커스텀 툴을 필요할 때 활용할 수 있기 때문에 llm자체에서 해결하지 못하는 약점을 보완할 수 있다.

    agent = initialize_agent(
        llm=llm, #쳇모델 설정
        verbose=True,  #동작 로그나 추가정보를 출력하여 동작 과정을 모니터링할 수 있게한다.
        agent= #어떤 agent로 초기화할지 정해줘야한다.
        tools=[ #사용하는 툴 등록
            StructuredTool.from_function(
                func=plus,
                name="Sum Calculator",
                description="Use this to perform sums of two numbers. This tool take two arguments, both should be numbers.",
            ),
        ],
    )

# 250123
    #12.2 How Do Agents Work
    StructuredTool : langchain은 llm이 Finish Agent 값을 반환하기 전까지 일정 형식에 맞춰 이전 기록과 함께 반복적으로 프롬프트를 던진다. >> 가끔 json이 아닌 text타입으로 반환하면 오류가 날 수 있다. >> functioncall을 이용하여 좀 더 답변의 틀을 강제하면 안정적으로 이용할 수 있다. >> #12.3내용
    https://python.langchain.com/v0.1/docs/modules/agents/concepts/

    #12.3 Zero-shot ReAct Agent
    Zero-shot ReAct Agent : functioncall을 이용하지 못하는 모델에도 적용가능하여 가장 범용적으로 사용된다. >> 단일 input만 받을 수 있다. 
    OpenAIFunctions : funcitoncall 기능을 이용하기 때문에 가장 안정적이다.

    #12.4 OpenAIFunctions Agent
    pydantic을 이용하여 우리의 함수와 입력값의 형태를 정의하도록 한다.

    ** type annotation 에러 발생 >> tool의 name/desciption 타입 추가
    PydanticUserError: Field 'name' defined on a base class was overridden by a non-annotated attribute. All field definitions, including overrides, require a type annotation.
    >>     
    class CalculatorTool(BaseTool):
        name = "CalculatorTool" >> name: str = "CalculatorTool"
        description = """~""" >> description: str = """~"""

    #12.5
    Duckduckgo 사용해서 주식 심볼 찾는 과정
    TypeError: DDGS.text() got an unexpected keyword argument 'max_results' 에러 발생
    >>pip install -U duckduckgo_search 설치 진행 후 VSC혹은 쥬피터 재실행하면 해결.

    #12.6
    alpha_vantage_api를 활용해서 툴을 늘리기

    #12.7
    stock agent streamlit에 옮기고 Agent가 OpenAI에 전달하는 system prompt를 커스터마이징

    #12.8
    FileManagementToolkit >> 파일을 다룰 수 있다.

# 240124
    #14.1
        https://platform.openai.com/assistants
        https://platform.openai.com/docs/assistants/overview
    #14.2
        OpenAI에서 제공하는 agent (Assisteant)를 만들어 보자.
            야후 파이낸스 설치 & OpenAi module을 최신으로 업데이트
            pip install yfinance openai --upgrade
        assistant생성
        assistant = client.beta.assistants.create(
            name="Investor Assistant",
            instructions="어시스턴트의 무드 설정",
            model="gpt-4-1106-preview",
            tools=[
                List형태의 문자열로 어떤 구조의 함수가 있는지 설명
            ],
        )

    #14.3 Assistant Tools
        쓰레드를 만들고 쓰레드에 메세지를 담은 뒤, 쓰레드를 런(실행)한다.

        DuckDuckGoSearchException: https://html.duckduckgo.com/html 202 Ratelimit
        >> 해결 못해서 return값을 고정하여 사용하도록 하고 다음 진도 진행.

        OpenAI assistant playground를 통해 어떤 입력값을 넣어야 할지 AI 답변 및 요청 내용이 있는지 확인할 수 있음.

    #14.4 Running A Thread
        쓰레드에 메세지를 담아서 만든다.
        만들어진 쓰레드를 어시스트를 지정하여 실행한다.
        run.status를 통해 in-progress / expired / require-action등의 응답이 오므로 해당 응답에 맞춰 메세지를 출력한다.

    #14.5 Assisstant Actions
        function_map을 이용해서 필요한 함수를 모아놓고 사용한다.
        어시스턴트에게 받은 질문 및 요청에 응답하기 위한 send_message함수


        send_message함수와 get_tool_outputs/ 를 통해 함수 응답결과를 반환한다.

    #14.8 RAG assistant >> 아직 다 안들음
        1. 어시스턴트 UI에서 직접 파일을 업로드 할 수 있다.
        2. 


# challenge 10
이유는 모르겠으나 json.loads()를 두번 먹이면 object변환이 된다..
    * 아무래도 ""가 안 밖으로 반복해서 나오는 것이 문제되는 듯.
    json.loads()를 두번하면 안쪽이 모두 ''로 바뀐다.
    a = json.loads(get_income_statement({"ticker": "AAPL"}))
    st.write(a) # 안됨
    st.write(json.loads(a)) # 변환됨