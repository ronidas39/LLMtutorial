from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

os.environ["OPENAI_API_KEY"] = "sk-proj-wbrNr25jaPp9JvUx73zqT3BlbkFJDsyo7Yc4VKmeagQV6gRR"
loader = PyPDFLoader(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial77\abc.pdf")

# loader=TextLoader(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial77\test.txt")
# documents=loader.load()
# text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
# docs=text_splitter.split_documents(documents)
# print(len(docs))

# for i in range(100):
#     vs=Chroma.from_documents(documents=docs,embedding=OpenAIEmbeddings(),persist_directory="./db_index_"+str(i))

vs=Chroma(persist_directory="./db_index",embedding_function=OpenAIEmbeddings())
info=vs.similarity_search("family of king john",k=3)
# print(info)