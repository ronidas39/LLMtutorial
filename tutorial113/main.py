import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
llm=ChatOpenAI(model="gpt-4o")

st.title("chat app with message history")
user=st.text_input("enter your username")
qsn=st.text_input("enter your question")
chat_with_history=MongoDBChatMessageHistory(
    session_id=user,
    connection_string="mongodb+srv://ronidas:llm1234@cluster0.lymvb.mongodb.net",
    database_name="langchain",
    collection_name="chat_history"
)
if user:
    if qsn:
        btn=st.button("ok")
        if btn:
            chat_with_history.add_user_message(qsn)
            response=llm.invoke(qsn)
            chat_with_history.add_ai_message(response.content)
            st.write(response.content)