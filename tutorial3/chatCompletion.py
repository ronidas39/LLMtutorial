from langchain.adapters import openai
msg=openai.ChatCompletion.create(messages=[
{"role":"system","content":"you are an intelligent assistant who can answer anything very smartly"},
{"role":'user',"content":"who won most ballon dor"}
]
)
print(msg)