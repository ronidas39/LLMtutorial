from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from pymongo import MongoClient
import urllib

llm=ChatOpenAI(model="gpt-4",temperature=0)
compressor=LLMChainExtractor.from_llm(llm)
embeddings=OpenAIEmbeddings()

client=MongoClient()
username="ronidas" 
pwd="pLy0SzzYX3zclsaq"

uri = "mongodb+srv://"+urllib.parse.quote_plus(username)+":"+urllib.parse.quote_plus(pwd)+"@cluster0.lymvb.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri)
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    collection=client["llm_tutorial"]["langchain"]
    vs=MongoDBAtlasVectorSearch(collection,embeddings,index_name="tutorial_35")
    docs=vs.max_marginal_relevance_search("what was the impact of covid on the world economy",k=1)
    print(docs[0].page_content)
    print("========================================================")
except Exception as e:
    print(e)
