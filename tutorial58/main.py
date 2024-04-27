from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st,os
import web_agent

st.title("talk with the rag")
st.write("ask anythng and get answer")
vectordb=Chroma(persist_directory=os.getcwd()+"/vector_index/",embedding_function=OpenAIEmbeddings())
prompt_template= """
you are an intelligent cyber security assistant who has 10 years of experience &  knowledge in sql injection
you task is to asnwer any question asked by user with the help of given context only nothing else ,
        {context}
        Question: {question}
        Answer:
        if you dont find any information within the context just reply with "NA"
"""
llm=ChatOpenAI(model="gpt-4-turbo",max_tokens=1024)
qa_chain=LLMChain(llm=llm,prompt=PromptTemplate.from_template(prompt_template))
question=st.text_area("ask your question here")
if question is not None:
    button=st.button("Submit")
    if button:
        rd=vectordb.similarity_search(question,k=3)
        context=""
        for d in rd:
            context += d.page_content
        response=qa_chain.invoke({"context":context,"question":question})
        result=response["text"]
        webagent_result=web_agent.runagent(question)
        if result=="NA":
            st.write(webagent_result)
        else:
            st.write(result)

