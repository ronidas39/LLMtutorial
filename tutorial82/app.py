from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import streamlit as st
import boto3

client=boto3.client("iam")
llm=ChatOpenAI(model="gpt-4o")
prompt="""
you are an intelligent assistant who is expert in extracting information from sentences , 
you can translate any {instruction}  into a json document to be used later,you must return the json and make e sure to use the below keys only :
username
password
follow the below conditions very strictly :
just include the json in the output nothing extra
"""
def genresponse(input):
    query_with_prompt=PromptTemplate(
        template=prompt,
        input_variables=["instruction"]
    )
    llmchain=LLMChain(llm=llm,prompt=query_with_prompt,verbose=True)
    response=llmchain.invoke(
        {"instruction":input}
                             )
    data=response["text"]
    data=data.replace("json","")
    data=data.replace("`","")
    data=json.loads(data)
    return data

st.set_page_config(page_title="AWS IAM AGENT")
st.header="what user you want to create"
input=st.text_input("write your task")
if input is not None:
    btn=st.button("submit")
    if btn:
        response=genresponse(input)
        st.write(response)
        user=client.create_user(UserName=response["username"])
        profile=client.create_login_profile(UserName=response["username"],Password=response["password"],PasswordResetRequired=False)
        st.write(profile)