#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

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

llm=ChatOpenAI(model="gpt-4o")
response=llm.invoke("what is cuurency for USA")
print(response)