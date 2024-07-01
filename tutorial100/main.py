from langchain_openai import ChatOpenAI
from langchain.agents import load_tools,initialize_agent,AgentType
llm=ChatOpenAI(temperature=0.0,model="gpt-4o")
tools=load_tools(["llm-math","wikipedia"],llm=llm)
agent=initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsingh_errors=True,
    verbose=True
)
agent("give some information on virat kohli Indian cricket player")