# challenge7
# QuizGPT를 구현하되 다음 기능을 추가합니다:

# 함수 호출(function calling.)을 사용합니다.
# 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
# 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
# 만점이면 st.ballons를 사용합니다.
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
# st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.


import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import json
from langchain.schema import BaseOutputParser, output_parser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)
output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters":{
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string"
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string"
                                    },
                                    "correct": {
                                        "type": "boolean"
                                    }
                                },
                                "required": ["answer", "correct"]
                            }
                        }
                },
                "required": ["question", "answers"]
            }
        }
    },
    "required": ["questions"]
 },#end of parameters
}#end of function

llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-3.5-turbo",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    #You can set the `function_call` arg to force the model to use a function
    function_call = {
      "name": "create_quiz",
    },
    functions=[function],
)

def  format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

question_prompt_hard = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
            Make 5 quizzes that very difficult about {subject}.
            each answers have four answer. one correct answer and three false answer.
        """
            )
    ]
)#end of prompt
question_prompt_medium = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
            Make 5 quizzes about {subject}.
            each answers have four answer. one correct answer and three false answer.
        """
            )
    ]
)#end of prompt

question_prompt_easy = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
            Make 5 quizzes that very easy about {subject}.
            each answers have four answer. one correct answer and three false answer.
        """
            )
    ]
)#end of prompt


@st.cache_data(show_spinner="Mainkg Quiz...")
def run_question_chain(subject, difficulty):
    # question_chain = {"context": format_docs} | question_prompt | llm
    if difficulty == "Easy":
        question_chain = {"subject": RunnablePassthrough()} | question_prompt_easy | llm
    elif difficulty == "Hard":
        question_chain = {"subject": RunnablePassthrough()} | question_prompt_medium | llm  
    else:
        question_chain = {"subject": RunnablePassthrough()} | question_prompt_medium | llm  
    
    return question_chain.invoke(subject)

@st.cache_data(show_spinner="Loading file...")
def spilt_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
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

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(topic):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(topic)

def painting_form():
    pass

with st.sidebar:
    topic = None
    docs = None
    difficulty = st.selectbox("Choose the difficulty of the quiz.", ["Easy", "Medium", "Hard"])
    
    # choice = st.selectbox(
    #     "Choose what you want to use.",
    #     ("File","Wikipedia Aricle")
    # )

    topic = st.text_input("Search Wikipedia Article")
    if topic:
        docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
                    
        Welcome to QuizGPT.
                    
        I will make 5 quiz from Wikipedia articles to test your knowledge.
                    
        Get started by searching on Wikipedia in the sidebar.
                    
    """
    )
else:
    response = run_question_chain(topic, difficulty)
    response_arguments = response.additional_kwargs["function_call"]["arguments"]
    response_arguments_json = json.loads(response_arguments)

    # if "quiz_answers" not in st.session_state:
    #     st.session_state["quiz_answers"] = {}

    with st.form("questions_form"):
        score = []
        for question in response_arguments_json["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Answers the question.",
                [answer["answer"] for answer in question["answers"]],
                index = None,
                # key=question["question"],
            )
            st.write(value)
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                score.append(1)
            elif value is not None:
                st.error("Wrong!")
                score.append(0)
        

        # st.write(score)
        # st.write(st.session_state)
        # for question in response_arguments_json["questions"]:
        #     st.write(st.session_state[question["question"]])
        
        
        button = st.form_submit_button("Submit Answers")
        if button:
            st.write(f"Your score is {sum(score)}")
            if sum(score) == 5:
                st.balloons()
            else:
                st.write("You can try again.")
                # 폼 리셋을 위해 상태 초기화
            restart = st.form_submit_button("Restart")
            st.session_state["quiz_answers"] = {}
            # st.write(st.session_state)
            
            if restart:
                st.session_state["quiz_answers"] = {}
