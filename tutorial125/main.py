from langchain_openai import ChatOpenAI
import asyncio
from browser_use import Agent
from browser_use.controller.service import Controller
import streamlit as st 

llm=ChatOpenAI(model="gpt-4o")
st.title("CHAT WITH ANY WEBSITE")
input=st.text_input("enter your question")

async def searchWeb(input):
    agent=Agent(task=input,llm=llm,controller=Controller(keep_open=False,headless=False))
    result=await agent.run()
    return result

if input is not None:
    btn=st.button("submit")
    if btn:
        result=asyncio.run(searchWeb(input))
        st.write(result.final_result())
