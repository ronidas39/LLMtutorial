
#pip3 install boto3
import boto3,json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from datetime import datetime
def update_table(item):
    client=boto3.resource("dynamodb")
    table=client.Table("token_logs")
    table.put_item(Item=item)
llm=ChatOpenAI(model="gpt-4o")
template="""you are intelligent assistant who can answer any {question} give by the user
response must be a json only , with the following keys,nothing else:
qsn:
ans:
"""
prompt=PromptTemplate.from_template(template)
chain=prompt|llm
response=chain.invoke({"question":"write a story on king james"})
answer=response.content
answer=answer.replace("json","")
answer=answer.replace("`","")
answer=json.loads(answer)
metadata=response.usage_metadata
metadata["response_id"]=response.id
metadata["qsn"]=answer["qsn"]
metadata["ans"]=answer["ans"]
metadata["timestamp"]=str(datetime.now())
update_table(metadata)