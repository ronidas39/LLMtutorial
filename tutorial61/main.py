from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from streamlit_mermaid import st_mermaid

def main():
    st.title("MERMAID DIAGRAM APP")
    st.write("create diagram from any details")
    qsn=st.text_area("ask your question here")
    ts="""
Your job is to write the code to generate a colorful mermaid diagram 
    describing the below
    {steps} information only , dont use any other information.
    only generate the code as output nothing extra.
    each line in the code must be terminated by ; 
    Code:
    """
    pt=ChatPromptTemplate.from_template(ts)
    llm=ChatOpenAI(model="gpt-4-turbo",temperature=0.0)
    qa_chain=LLMChain(llm=llm,prompt=pt)
    if qsn is not None:
        btn=st.button("submit")
        if btn:
            response=qa_chain.invoke({"steps":qsn})
            data=response["text"]
            data=data.replace("`","")
            data=data.replace("mermaid","")
            st_mermaid(data,key="flow",height="600px")
            st.text(data)


if __name__ == "__main__":
    main()