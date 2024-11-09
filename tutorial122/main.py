from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv()
llm=ChatOpenAI(model="gpt-4o")
tool=TavilySearchResults(
    max_results=1,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True
)
graph=Neo4jGraph()
llm_transformer=LLMGraphTransformer(llm=llm)
prompt=ChatPromptTemplate.from_template("summarize this content with minimum sentences with focusing on all important information {context}")
response=tool.invoke({"query":"apple new research news"})
raw_content=response[0]["content"]
chain=create_stuff_documents_chain(llm,prompt)
sd=[]
sd.append(Document(page_content=raw_content,metadata={"source":response[0]["url"]}))
summary=chain.invoke({"context":sd})
gd=[]
gd.append(Document(page_content=summary,metadata={"source":response[0]["url"]}))
graph_docs=llm_transformer.convert_to_graph_documents(gd)
graph.add_graph_documents(graph_docs)
