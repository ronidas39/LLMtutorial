from langchain.callbacks import FileCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType,initialize_agent,load_tools
import os
from loguru import logger

os.environ["SERPAPI_API_KEY"]="c0e23b261ee29de6ab6155ca25f0e8845a9a8e1a60d2fd62bd1d95d9f8772bc6"
llm=ChatOpenAI(temperature=0,model="gpt-4")
tools=load_tools(["serpapi"],llm=llm)
logfile="output.txt"


handler=FileCallbackHandler(logfile)
logger.add(logfile,colorize=True,enqueue=True)
agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handler=[handler],verbose=True)
result=agent.run("who is the current prime minister of Australia")
logger.info(result)