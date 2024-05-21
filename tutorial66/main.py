import streamlit as st
from pymongo import MongoClient
import json,urllib
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from ticket import generate_ticket



llm=ChatOpenAI(model="gpt-4o",temperature=0.0)
prompt="""
you are an intelligent assistant who is expert in understanding user problem  ,
you can translate any {instruction}  into a json document to be used to create jira service ticket.
While returning the json make sure to use the below keys only :

Title
Summary

follow the below conditions very strictly :
just include the json in the output nothing extra
"""

def genresponse(input):
    query_with_prompt=PromptTemplate(
        template=prompt,
        input_variables=["instruction"]
    )
    llmchain=LLMChain(llm=llm,prompt=query_with_prompt,verbose=True)
    response=llmchain.invoke({
        "instruction":input
    })
    data=response["text"]
    data=data.replace("json","")
    data=data.replace("`","")
    data=json.loads(data)
    return data


st.title("AI driven HRMS app")
st.write("enter your instructions in english")
input=st.text_area("write your instruction")
if input is not None:
    btn=st.button("submit")
    if btn:
        response=genresponse(input)
        st.write(response)
        
            