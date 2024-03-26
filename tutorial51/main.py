from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import streamlit as st
import boto3

llm=ChatOpenAI(model="gpt-4",temperature=0.0)
client=boto3.client("s3",region_name="us-east-1")
st.set_page_config(page_title="AWS GPT")
st.header="ask anything related to aws"
input=st.text_input("what you want me to do")

def getS3BucketName(input):
    names=[]
    buckets=client.list_buckets()["Buckets"]
    for bucket in buckets:
        names.append(bucket["Name"])
    return(names)

def createS3Bucket(input):
    response=client.create_bucket(Bucket=input)
    data=response["ResponseMetadata"]["HTTPStatusCode"]
    if data==200:
        return "successfully created"
    else:
        return "failed to create"


s3name=Tool(
    name="BucketNames",
    func=getS3BucketName,
    description="use to get s3 bucket names using BucketNames tool"
)


s3create=Tool(
    name="BucketCreate",
    func=createS3Bucket,
    description="use to create s3 bucket using BucketCreate tool as per the input name provided for s3 bucket by the end user"
)
tools=[s3name,s3create]
agent=initialize_agent(tools,llm,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
if st.button("Submit",type="primary"):
    if input is not None:
        response=agent.run(input)
        st.write(response)