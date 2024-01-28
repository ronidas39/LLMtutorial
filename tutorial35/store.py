from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import urllib

url="https://en.wikipedia.org/wiki/COVID-19"
loader=WebBaseLoader([url])
doc=loader.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0,separators=["\n\n","\n","(?<=\.)"," "],length_function=len)
docs=splitter.split_documents(doc)
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
    docsearch=MongoDBAtlasVectorSearch.from_documents(docs,embeddings,collection=collection,index_name="tutorial_35")
    print("documents are stored into mongodb vector store successfully")
    
    
except Exception as e:
    print(e)