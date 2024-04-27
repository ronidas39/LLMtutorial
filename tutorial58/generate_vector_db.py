import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 

def gendb() -> Chroma:
    folder=os.getcwd()
    documents=[]
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            pdf_file=os.path.join(folder,file)
            print(pdf_file)
            loader=PyPDFLoader(pdf_file)
            documents.extend(loader.load())
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=10)
    cd=text_splitter.split_documents(documents)
    vectordb=Chroma.from_documents(
        documents=cd,
        embedding=OpenAIEmbeddings(),
        persist_directory=os.getcwd()+"/vector_index/"
    )
    vectordb.persist()

gendb()