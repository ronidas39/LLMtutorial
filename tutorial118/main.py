from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from genImage import genImage
from getImageUrl import getUrl


load_dotenv()
auth_token=os.getenv("auth_token")

llm=ChatOpenAI(model="gpt-4o")
template="""
As an AI catalog assistant specialized in interior design, 
your goal is to understand and respond to user requests {question} about designing, decorating, and furnishing interiors.
For every relevant request, interpret the design style, ambiance, materials, colors, and layout based on the user’s question and 
generate a detailed description only that represents the requested interior concept. 
Later designer will use the description to create The images ,imageshould look visually appealing, fit a catalog setting,
and reflect modern design standards.
You answer must within short one or two line but effective

If the user’s question is unrelated to interior design, respond with 'I don't know.
"""
prompt=PromptTemplate.from_template(template)
chain=prompt|llm
response=chain.invoke({"question":"modern office room with all accessiories"})
ids=response.content
genid=genImage(ids,auth_token)
urldata=getUrl(genid,auth_token)
print(urldata)
