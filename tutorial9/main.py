from langchain.agents import AgentType,Tool, initialize_agent,AgentExecutor
from langchain.llms import OpenAI
import os
from langchain.utilities import SerpAPIWrapper

os.environ["SERPAPI_API_KEY"]="xxxxxxxxxxxxxxxxxxxx"
AgentExecutor.handle_parsing_errors=True
llm=OpenAI(temperature=0.0)
search=SerpAPIWrapper()
tools=[
    Tool(
        name="Intermediate Answer",func=search.run,description="you can perform any search"
    )
]
search_agent=initialize_agent(tools,llm=llm,agent=AgentType.SELF_ASK_WITH_SEARCH,verbose=True)
print(search_agent.run("who was the prime minister of India in 2001"))
