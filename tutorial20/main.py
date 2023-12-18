from langchain.agents import initialize_agent,AgentType,load_tools
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_verbose
set_verbose(True)

llm=ChatOpenAI(temperature=0,model="gpt-4")
tools=load_tools(["ddg-search"],llm=llm)
agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
response=agent.run("Give me information on the crypto market")
print(response)