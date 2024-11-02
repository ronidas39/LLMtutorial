import io
from langsmith import Client
import os
from dotenv import load_dotenv
load_dotenv()
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
client=Client()
with io.open("data.csv","r",encoding="utf-8")as f1:
    data=f1.read()
    f1.close()
lines=data.split("\n")[1:]
dataset_name="test_data"
dataset=client.create_dataset(dataset_name=dataset_name)
qsns=[]
answs=[]
for line in lines:
    qsn=line.split(",")[0]
    qsns.append({"question":qsn})
    ans=line.split(",")[1]
    answs.append({"answer":ans})

client.create_examples(
    inputs=qsns,
    outputs=answs,
    dataset_id=dataset.id
)


