import boto3
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

client=boto3.resource("dynamodb",region_name="us-east-1")
table=client.Table("token_logs")

llm=ChatOpenAI(model="gpt-4o")
template="""
you are intelligent assistant who can write any article on {topic} with nice and attractive lines.
output will only article nothinbg extra
"""
prompt=PromptTemplate.from_template(template)
chain=prompt|llm
response=chain.invoke({"topic":"Indian politics"})
answer=response.content
response_id=response.id
data=response.usage_metadata
data.update({"response_id":response_id,"answer":response.content,"question":template})
table.put_item(Item=data)

