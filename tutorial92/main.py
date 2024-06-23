from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json


loader=PyPDFLoader(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial92\employee_info.pdf")
docs=loader.load()

llm=ChatOpenAI(model="gpt-4o")
template="""
you are an intelliegnt bot who can analyze any text with  employee information{doc_text} , 
your job is to read and analyse the information and create a json dictionary.
dictionary has the followinmg key:
Name:
Employee Id:
City:
Department:
Company:
Skills:
Experience:
Dob:
Salary:

output must be json nothing else
"""
prompt=PromptTemplate(template=template,input_variables=["doc_text"])
llmchain=LLMChain(llm=llm,prompt=prompt)
emp=[]
for doc in docs:
    response=llmchain.invoke({"doc_text":doc.page_content})
    data=response["text"]
    data=data.replace("json","")
    data=data.replace("`","")
    data=json.loads(data)
    emp.append(data)
print(emp)
    