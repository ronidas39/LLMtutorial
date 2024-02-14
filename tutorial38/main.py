from langchain.agents import initialize_agent,AgentType,load_tools
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
os.environ["SERPAPI_API_KEY"]="c0e23b261ee29de6ab6155ca25f0e8845a9a8e1a60d2fd62bd1d95d9f8772bc6"
tools=load_tools(["serpapi"])

ts="""
You are an intelligent search master and analyst who can search internet using serpapi tool and analyse any product to find the brand of the product ,name of the product,
product description,price and rating between 1-5 based on your owen analysis.
Take the input below delimited by tripe backticks and use it to search and analyse using serapi tool
input:```{input}```
then based on the input you format the output as JSON with the following keys:
brand_name
product_name
description
price
rating
"""
pt=ChatPromptTemplate.from_template(ts)
llm=ChatOpenAI(model="gpt-4",temperature=0.0)
agent=initialize_agent(tools,llm,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
pi=pt.format_messages(input="best sports car in Germany")
pa_response=agent.run(pi)
print(type(pa_response))