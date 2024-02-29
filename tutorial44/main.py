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

graph=Neo4jGraph(url=url,username=username,password=password)