from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


loader=PyPDFLoader("C:\\Users\\welcome\\OneDrive\\Documents\\GitHub\LLMtutorial\\tutorial67\\manual.pdf")
docs=loader.load()
print(docs)
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
chunked_documents=text_splitter.split_documents(docs)
vectordb=Chroma.from_documents(
    docs, OpenAIEmbeddings(), persist_directory="./chroma_db"
)
