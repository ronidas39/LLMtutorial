from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM,OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.prompts import PromptTemplate
llm=OllamaLLM(model="gemma3:27b")
embedding=OllamaEmbeddings(model="llama3.2:3b")
template="""
you are an intelligent assistant for question-answering task
here your task is to answer the user question based on the following provided retrieved context only
if you dont know the answer just say you dont know
make the answer very professional and decsriptive and explanation driven
question:{qsn}
context:{context}
"""
prompt=PromptTemplate.from_template(template)
chain=prompt|llm
vs=Chroma(persist_directory="./index",embedding_function=embedding)
st.title("multimodal rag")
qsn=st.text_input("ask your question")
if qsn is not None:
    btn=st.button("submit")
    if btn:
        context_docs=vs.similarity_search(qsn,k=3)
        context=""
        images=[]
        for doc in context_docs:
            context=context+doc.page_content
            if doc.metadata["source"]=="image-content":
                images.append(doc.metadata["name"])
        response=chain.invoke({"context":context,"qsn":qsn})
        st.write(response)
        for image in images:
            st.image(image)

