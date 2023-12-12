from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType,initialize_agent,load_tools
import os
os.environ["SERPAPI_API_KEY"]="c6083f1c51726d299bb135d9c780c9927758e8f48a29d55a8be6723f842d18d2"
llm=ChatOpenAI(temperature=0,model="gpt-4")
tools=load_tools(["serpapi"],llm=llm)
agent=initialize_agent(tools,llm,agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,return_intermediate_steps=True)
response=agent({"input":"who is the present prime minister of India,what is his age"})
print(response)