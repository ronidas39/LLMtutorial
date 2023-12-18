from langchain.agents import initialize_agent,AgentType,load_tools
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_debug
set_debug(True)

llm=ChatOpenAI(temperature=0,model="gpt-4")
tools=load_tools(["ddg-search"],llm=llm)
agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
agent.run("what is the current weather conditions in India accross different top cities")