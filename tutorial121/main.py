from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
documents=[]
llm=ChatOpenAI(model="gpt-4o")
load_dotenv()
graph=Neo4jGraph()
llm_transformers=LLMGraphTransformer(llm=llm)

loader=PyPDFLoader(file_path=r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial121\cv1.pdf")
docs=loader.load()
source=docs[0].metadata["source"]
prompt=ChatPromptTemplate.from_template(
           """Summarize this content with 4-5 sentences focusing on personal infomation,education,organization,experience in organization,
               project details,skills,programming language :{context}
                                        """)
chain=create_stuff_documents_chain(llm,prompt)
response=chain.invoke({"context":docs})
summary=Document(page_content=response,metadata={"source":source})
documents.append(summary)
graph_documents=llm_transformers.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents)


