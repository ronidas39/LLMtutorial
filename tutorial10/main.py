from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
history=DynamoDBChatMessageHistory(table_name="msgTable",session_id="1")
history.add_user_message("hi")
history.add_ai_message("i am doing well")
print(history.messages)