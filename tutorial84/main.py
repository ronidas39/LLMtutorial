from bs4 import BeautifulSoup
import json
import xmltodict
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec
from langchain_openai import ChatOpenAI


with open(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial84\sample.xml","r") as f1:
    content=f1.read()
text_content=str(BeautifulSoup(content,"lxml"))
xml_dict=xmltodict.parse(text_content)
spec=JsonSpec(dict_=xml_dict)
toolkit=JsonToolkit(spec=spec)
agent=create_json_agent(llm=ChatOpenAI(temperature=0,model="gpt-4o"),toolkit=toolkit,max_iterations=3000,verbose=True)
response=agent.run("Please give all application names ,you will find it inside application key , it congtains a list")
print(response)