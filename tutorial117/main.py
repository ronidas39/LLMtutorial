import io
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(model="gpt-4o")
import os,glob
cwd=os.getcwd()
files=glob.glob(cwd+"/*.txt")
documents=[]
for file in files:
    with io.open(file,"r",encoding="utf-8")as f1:
        data=f1.read()
        f1.close()
    documents.append(Document(page_content=data,metadata={"title":file.split("\\")[-1]}))
# print(documents)

prompt=ChatPromptTemplate.from_template("Summarize this content with little bullet :{context}")
chain=create_stuff_documents_chain(llm,prompt)
response=chain.invoke({"context":documents})
print(response)
    

