from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
llm=ChatOpenAI(model="gpt-4o")
loader=TextLoader(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial78\test.txt")
docs=loader.load()
text_splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=0)
document=text_splitter.split_documents(docs)

URL = "https://test-2lt4id2i.weaviate.network"
APIKEY = "DnmmkH4txFMUr6UnAg0SCBbOSeuzaLANxhqF"
  
# Connect to a WCS instance
client = weaviate.connect_to_wcs(
    cluster_url=URL,
    auth_credentials=weaviate.auth.AuthApiKey(APIKEY))
# vs=WeaviateVectorStore.from_documents(document,embedding=OpenAIEmbeddings(),client=client)
vs=WeaviateVectorStore(client=client,index_name="LangChain_65700351ec2b4baf889198b62eeb6e13",embedding=OpenAIEmbeddings(),text_key="text")
retriever=vs.as_retriever()
template= """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use five sentences minimum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt=ChatPromptTemplate.from_template(template)
rag_chain=(
    {"context":retriever,"question":RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)
response=rag_chain.invoke("write is the moral of this story")
print(response)
