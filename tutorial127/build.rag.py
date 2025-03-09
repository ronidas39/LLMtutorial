from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(model="gpt-4o")
ts=RecursiveCharacterTextSplitter(chunk_size=602,chunk_overlap=0)
embeddings=OpenAIEmbeddings()
loader=TextLoader(file_path="book.txt")
docs=loader.load()
sd=ts.split_documents(docs)
vs=Chroma.from_documents(sd,embedding=embeddings,persist_directory="./index")
