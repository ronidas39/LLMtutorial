from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent
import pandas as pd
agent=create_csv_agent(llm=OpenAI(temperature=0),path=r"C:\Users\welcome\Documents\GitHub\LLMtutorial\tutorial4\books.csv",verbose=True,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
print(agent.run("find the author who has most number of books published in year 2015"))