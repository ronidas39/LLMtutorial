import os
from langchain.agents import AgentType,initialize_agent,load_tools
from langchain.chat_models import ChatOpenAI
from langchain.utilities import Portkey
os.environ["SERPAPI_API_KEY"]="xxx"
port_key="86ipO9H+YMA/xxxxxx="
TREACE_ID="TUTORIAL-15"
headers=Portkey.Config(
    api_key=port_key,
    trace_id=TREACE_ID
)
llm=ChatOpenAI(temperature=0,model="gpt-4",headers=headers)
tools=load_tools(["serpapi","llm-math"],llm=llm)
agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
agent.run("what was the coldest day in 2022 in New Delhi,mention the date and  temparature in celcius raiserd to the power of .05")