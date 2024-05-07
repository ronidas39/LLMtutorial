from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings
import os

loader=PyPDFLoader("vedas.pdf")
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
chunked_documents=text_splitter.split_documents(docs)
vectordb=Chroma.from_documents(
    documents=chunked_documents,
    embedding=OpenAIEmbeddings(),
    persist_directory=os.getcwd()+"/vector_index"
)
vectordb.persist()