import boto3
from langchain_aws import ChatBedrock
client=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
llm=ChatBedrock(model_id="amazon.titan-text-premier-v1:0",client=client)
response=llm.invoke("write an articvle on AI in Education sector")
print(response.content)