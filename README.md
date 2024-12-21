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