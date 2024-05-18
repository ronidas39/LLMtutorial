import streamlit as st
from pymongo import MongoClient
import json,urllib
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

username="ronidas"
pwd="t2HKvnxjL38QGV3D"
client=MongoClient("mongodb+srv://"+urllib.parse.quote(username)+":"+urllib.parse.quote(pwd)+"@cluster0.lymvb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db=client["employee"]
collection=db["employee"]

llm=ChatOpenAI(model="gpt-4o",temperature=0.0)
prompt="""
you are an intelligent assistant who is expert in mongodb , you can translate any {instruction}  into a json document to be uploaded into mongodb.While returning the json make sure to use the below keys only :
FirstName
LastName
Age
Sex
Country
City
Department
Designation
Dob
SSN

follow the below conditions very strictly :
just include the json in the output nothing extra
"""
def check_mandatory_keys(data):
    missing_keys=[]
    mandatory_keys=["FirstName","LastName","SSN"]
    for key in mandatory_keys:
        if not data[key]:
            missing_keys.append(key)
    return missing_keys

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
        missing_check=check_mandatory_keys(response)
        if missing_check:
            st.text(f"{missing_check} is/are empty")
        else:
            insert_object=collection.insert_one(response)
            st.text(insert_object.inserted_id)
