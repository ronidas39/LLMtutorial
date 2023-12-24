from langchain.agents import AgentType,initialize_agent,load_tools
from langchain.chat_models import ChatOpenAI
llm=ChatOpenAI(temperature=0,model="gpt-4")
tools=load_tools(["ddg-search"],llm=llm)
agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
agent.run("top 5 business use case on LLM and generative ai")