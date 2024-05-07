from langchain_community.vectorstores import Chroma 
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import os
st.title("TALK TO ME")
st.write("promting example with rag")

vectordb=Chroma(persist_directory=os.getcwd()+"/vector_index",embedding_function=OpenAIEmbeddings())
prompt_template="""
You are very smart AI assistant who is very good in explaining and expressing answer for anything asked by the user
user will ask question as {question}
you must give answer with detailed explanation with bullets, heading , subheading etc. based on the {context} only
always rewrite  the title ,headings, subheading in your ,try to avoid section numbers chapter numbers in the title or heading dont keep the same as the given in the {context} but keep the meaning same.
you must follow this  very stricty , dont use anything else other than the given {context}
if no related information found from the {context} just reply with "I dont know", this is very important
answer:
answer:
"""
llm=ChatOpenAI(model="gpt-4-turbo",max_tokens=1024)
qa_chain=LLMChain(llm=llm,prompt=PromptTemplate.from_template(prompt_template))
question=st.text_area("ask your question")

if question is not None:
    button=st.button("submit")
    if button:
        rd=vectordb.similarity_search(question,k=12)
        context=""
        for d in rd:
            context +=d.page_content
        response=qa_chain.invoke({"question":question,"context":context})
        result=response["text"]
        st.write(result)



