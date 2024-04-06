from llmload import loadllm
import requests
from langchain.agents import AgentType,initialize_agent
from langchain.tools import Tool

def getPrice(input):
    url="https://api.coincap.io/v2/assets/"+input.lower()
    response=requests.get(url).json()
    price=response["data"]["priceUsd"]
    return price

apicall=Tool(
    name="getCryptoPrice",
    func=getPrice,
    description="use to get the price for any given crypto from user input"

)
tools=[apicall]
llm=loadllm()
agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
print(agent.run("what is the price of shiba-inu"))