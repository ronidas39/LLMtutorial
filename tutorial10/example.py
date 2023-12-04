from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.adapters import openai
msgs=["what is crypto currency","is programming required to learn blockchain","what is the future of blockchain"]
i=2
for msg in msgs:
    history=DynamoDBChatMessageHistory(table_name="msgTable",session_id=str(i))

    ai_msg=openai.ChatCompletion.create(messages=[
        {"role":"system","content":"you are an intelligent assistant who can answer anything with intelligence"},
        {"role":"user","content":msg}
    ])
    history.add_user_message(msg)
    history.add_ai_message(ai_msg["choices"][0]["message"]["content"])
    i=i+1
