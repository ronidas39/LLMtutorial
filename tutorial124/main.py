from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
load_dotenv()
graph=Neo4jGraph()
schema=graph.get_schema
llm=ChatOpenAI(model="gpt-4o")
template="""
Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided nodes, relationship and properties from the schema provided below:
{schema}
Do not use any other relationship types or properties that are not provided.
sample question and queries given below:
question:display the os name which is used maximum time
query:MATCH (:Machine)-[:RUNS]->(os:OS) RETURN os.name AS osName ,count(os) as os order by os DESC limit 1
The question is:
{question}
your response will be only json with below key nothing else
'query':
"""
prompt=PromptTemplate.from_template(template)
chain=prompt | llm
response=chain.invoke({"schema":schema,"question":"display all racks name and count of machine it is holding"})
response=response.content
response=response.replace("json","")
response=response.replace("`","")
query=json.loads(response)["query"]
print(query)
result=graph.query(query=query)
print(result)