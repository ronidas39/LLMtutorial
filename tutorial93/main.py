from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
import json


llm=ChatOpenAI(model="gpt-4o")
loader=PyPDFLoader(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial93\Hospital_Report.pdf")
docs=loader.load()
template="""
you are an intelliegnt bot who can analyze any text with  hospital information{doc_text} , 
your job is to read and analyse the information and create a json dictionary.
dictionary has the followinmg key:
Hospital Name
Address
City
State
Zip Code
Contact Number
Email Address
Website

output must be json nothing else
"""
prompt=PromptTemplate(template=template,input_variables=["doc_text"])
llmchain=LLMChain(llm=llm,prompt=prompt)
for doc in docs:
    text=doc.page_content
    response=llmchain.invoke({"doc_text":text})
    data=response["text"]
    data=data.replace("json","")
    data=data.replace("`","")
    data=json.loads(data)
    print(data)

