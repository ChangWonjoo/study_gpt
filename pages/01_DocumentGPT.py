import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("DocumentGPT")

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save == True:
        st.session_state["messages"].append({"message": message, "role": role})

if "messages" not in st.session_state:
    st.session_state["messages"] = []
else:
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)
# st.write(st.session_state)

message = st.chat_input("Send a message to the AI")

if message:
    send_message(message, "human")
    time.sleep(2)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)