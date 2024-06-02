from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","what are everyone's age:\n\n{context}")
    ]
)
llm=ChatOpenAI(model="gpt-4o")
chain=create_stuff_documents_chain(llm,prompt)
loader=DirectoryLoader(path=os.getcwd(),glob="**/*.txt")
docs=loader.load()
print(len(docs))
response=chain.invoke({"context":docs})
print(response)