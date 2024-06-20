from langchain.agents import initialize_agent,AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
tools=load_tools(["serpapi"])
ts="""
You are an intelligent search master and analyst who can search internet using serpapi tool and analyse information about any company to find the following information:
Company name,Founder,Ceo,Company Details,Prdoucts,Phone number,Email,Company website,Address,City,Postal Code

Take the input below delimited by triple backticks and use it to search and analyse using serapi tool
input:```{input}```
then based on the input you have to generate final output with the following keys:
Company name
Founder
Ceo
Company Details
Prdoucts
Phone number
Email
Company website
Address
City
Postal Code
"""
pt=ChatPromptTemplate.from_template(ts)
llm=ChatOpenAI(model="gpt-4o",temperature=0)
agent=initialize_agent(tools,llm,agernt_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,agent_kwargs={"handle_parsing_errorts":True})
agent.handle_parsing_errors=True
st.title("COMPANY INFORMATION FINDER APP")
input=st.text_input("write the company name")
if input is not None:
    btn=st.button("submit")
    if btn:
        pi=pt.format_messages(input=input)
        response=agent.run(pi)
        st.write(response)