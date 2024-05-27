from langchain_community.vectorstores import Chroma 
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import os
st.title("help desk agent for printers")
st.write("printer help desk")

vectordb=Chroma(persist_directory=os.getcwd()+"/chroma_db",embedding_function=OpenAIEmbeddings())
prompt_template="""
You are an intelligent help desk agent specializing in printers. 
Answer user queries {input} related to printers based strictly on the provided {context}. 
Do not make assumptions or provide information not included in the {context}.
"""
llm=ChatOpenAI(model="gpt-4o",max_tokens=1024)
qa_chain=LLMChain(llm=llm,prompt=PromptTemplate.from_template(prompt_template))
question=st.text_area("ask your question")

if question is not None:
    button=st.button("submit")
    if button:
        rd=vectordb.similarity_search(question,k=5)
        context=""
        for d in rd:
            context +=d.page_content
        print(context)
        response=qa_chain.invoke({"input":question,"context":context})
        result=response["text"]
        print(result)
        st.write(result)