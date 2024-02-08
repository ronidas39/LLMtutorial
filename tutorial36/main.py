from langchain.agents import AgentType,initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import requests


def getPrice(input):
    url="https://api.coincap.io/v2/assets/"+input.lower()
    response=requests.get(url)
    price=response.json()["data"]["priceUsd"]
    return price

llm=ChatOpenAI(model="gpt-4",temperature=0.0)
apicall=Tool(
    name="getCryptoPrice",
    func=getPrice,
    description="use to get the price for any given crypto from user input"
)
tools=[apicall]
agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
print(agent.run("what is the price of cardano"))