from langchain.agents.agent_toolkits import GmailToolkit
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType,initialize_agent

toolkit=GmailToolkit()

llm=ChatOpenAI(temperature=0.0,model="gpt-4")
agent=initialize_agent(tools=toolkit.get_tools(),llm=llm,verbose=True,max_iterations=1000,max_execution_time=1600,agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
print(agent.run("how many emails received on avergare daily basis , check for last 4 days"))