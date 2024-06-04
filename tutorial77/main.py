from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# loader=TextLoader(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial77\test.txt")
# documents=loader.load()
# text_splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=0)
# docs=text_splitter.split_documents(documents)
# vs=Chroma.from_documents(documents=docs,embedding=OpenAIEmbeddings(),persist_directory="./db_index")

vs=Chroma(persist_directory="./db_index",embedding_function=OpenAIEmbeddings())
info=vs.similarity_search("family of king john",k=3)
print(info)