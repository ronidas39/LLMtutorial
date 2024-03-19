from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WikipediaLoader
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

load_dotenv()
diffbot_api_key=os.getenv("API_KEY")
url="bolt://3.83.139.248:7687"
username="neo4j"
password="deputy-compensations-defaults"
nlp=DiffbotGraphTransformer(diffbot_api_key=diffbot_api_key)

graph=Neo4jGraph(url=url,username=username,password=password)

raw_docs=WikipediaLoader(query="Narendra_Modi").load()
graph_documents=nlp.convert_to_graph_documents(raw_docs)
print(graph_documents)
# graph.add_graph_documents(graph_documents)

# chain=GraphCypherQAChain.from_llm(
#     cypher_llm=ChatOpenAI(model_name="gpt-4",temperature=0.0),
#     qa_llm=ChatOpenAI(model_name="gpt-4",temperature=0.0),
#     graph=graph,
#     verbose=True
# )
# response=chain.run("what is the Nationality of Narendra Modi")
# print(response)