from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

loader=TextLoader("review.txt")
docs=loader.load()
llm=ChatOpenAI(temperature=0,model="gpt-4-turbo-preview")
chain=load_summarize_chain(llm,chain_type="stuff")
