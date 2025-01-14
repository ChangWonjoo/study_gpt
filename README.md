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