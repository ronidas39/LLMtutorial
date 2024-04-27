from langchain.agents import initialize_agent,AgentType, load_tools
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

os.environ["SERPAPI_API_KEY"]="d71fed0327d76cf827022dd391bf23ca7392d5632da3bf8014ba736e25434233"
tools=load_tools(["serpapi"])
ts="""
You are an  intelligent search master and analyst who can search internet using serpapi tool and 
analyse and generate accurate answer with required explanation
Take the input below delimited by tripe backticks and use it to search and analyse using serapi tool
answer should well explained and must be written in simple way 
this answers will be used by students
input:```{input}```
make sure to geneate the results in english
"""
pt=ChatPromptTemplate.from_template(ts)
llm=ChatOpenAI(model="gpt-4-turbo",temperature=0)
agent=initialize_agent(tools,llm,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

def runagent(input):
    prompt=pt.format_messages(input=input)
    response=agent.run(prompt)
    return response