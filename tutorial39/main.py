from langchain.agents import AgentType,initialize_agent,load_tools
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema,StructuredOutputParser
from langchain_openai import ChatOpenAI
import os
os.environ["SERPAPI_API_KEY"]="4132eb5fa197a54daf77381f7441a4b44c916b7e39e58cf4221f6797f29d083d"
tools=load_tools(["serpapi"])
llm=ChatOpenAI(model="gpt-4",temperature=0.0)
brand_name=ResponseSchema(name="brand_name",description="this is the brand of the product")
product_name=ResponseSchema(name="product_name",description="this is the product name")
description=ResponseSchema(name="description",description="this is the short description of the product")
product_price=ResponseSchema(name="price",description="this will be in number, represents the price of the product")
product_rating=ResponseSchema(name="rating",description="this is whole integer,this gives the rating between 1-10")
response_schema=[brand_name,product_name,description,product_price,product_rating]
output_parser=StructuredOutputParser.from_response_schemas(response_schema)
format_instruction=output_parser.get_format_instructions()
ts="""
You are an intelligent search master and analyst who can search internet using serpapi tool and analyse any product to find the brand of the product ,name of the product,
product description,price and rating between 1-5 based on your owen analysis.
Take the input below delimited by tripe backticks and use it to search and analyse using serapi tool
input:```{input}```
{format_instruction}
"""
prompt=ChatPromptTemplate.from_template(ts)
fs=prompt.format_messages(input="best android phone in India",format_instruction=format_instruction)
agent=initialize_agent(tools,llm,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
response=agent.run(fs)
output=output_parser.parse(response)
print(output["brand_name"],output["product_name"])