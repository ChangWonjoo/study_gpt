# study_gpt
for studying LLM AI with nomad coder..


# 241215
    1. codespace 설치 후 확인해보니, python 3.12로 설치 되어있어, 설정 바꾸려니 복잡해보여서 로컬로 먼저 해보기로 함.

    2. 로컬 PC도 3.12로 되어있어서 3.11.9버젼 설치 진행
    https://www.python.org/downloads/release/python-3119/

    3. python3.11 -m venv ./env && python3.11 --version 코멘드 찾을 수 없다는 에러 발생 (패스 설정 문제일듯?)
    py -3.11 -m venv ./env 로 env생성 완료.
    >> ./env/Scripts/activate로 가상환경 실행 (나갈 때는 deactivate)

    4. pip install -r requirements.txt >> 설치 확인용 main.py로 ticktoken 라이브러리 위치 확인되면 완료(<module 'tiktoken' from 'C:\\Users\\wonjooLAPTOP\\Dropbox\\github\\study_gpt\\env\\Lib\\site-packages\\tiktoken\\__init__.py'>).

    5. .env 파일 생성 후 설정 값 작성.