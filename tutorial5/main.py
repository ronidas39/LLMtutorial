import streamlit as st
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd


st.set_page_config(page_title="AI APP TO TALK WITH CSV")
st.header="ASK ANYTHING"

csv_file=st.file_uploader("upload your csv file",type="csv")
if csv_file is not None:
    df=pd.read_csv("books.csv")
    agent=create_pandas_dataframe_agent(llm=OpenAI(temperature=0),df=df,max_execution_time=1600,max_iterations=1000,verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    query=st.text_input("ask question to your csv")

    if st.button("Submit",type="primary"):
        if query is not None:
            response=agent.run(query)
            st.write(response)