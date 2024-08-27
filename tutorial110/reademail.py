from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
llm=ChatOpenAI(model="gpt-4o")
template="""
you are intelligent assistant , who can read any group of {text} related one email and identify the body of the email,
then create a json with the following key only,
"username":
output must be only json nothing extra
"""
prompt=PromptTemplate.from_template(template)
chain=prompt|llm

def readbody(text):
    response=chain.invoke({"text":text})
    data=response.content
    data=data.replace("json","")
    data=data.replace("`","")
    data=json.loads(data)
    return data


