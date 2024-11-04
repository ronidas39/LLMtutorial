import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from genScript import genText
from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(model="gpt-4o")
llm_transformer=LLMGraphTransformer(llm=llm)
graph=Neo4jGraph()
documents=genText("https://www.youtube.com/watch?v=HxTNuGnYZWM")
graph_documents=llm_transformer.convert_to_graph_documents(documents)
# print(f"Nodes:{graph_documents[0].nodes}")
# print(f"Nodes:{graph_documents[0].relationships}")
graph.add_graph_documents(graph_documents)