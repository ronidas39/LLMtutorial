from langchain.agents import AgentType,initialize_agent,load_tools
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import FileManagementToolkit
import os

wd=os.getcwd()
llm=ChatOpenAI(temperature=0.0)
toolkit=FileManagementToolkit(root_dir=wd)
tools=toolkit.get_tools()
agent=initialize_agent(tools,llm,agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=True,
                       agent_executor_kwards={"handle_parsing_erros":True})
print(agent.run("delete all files with .csv extension"))