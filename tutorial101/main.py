from langchain_community.llms import Ollama

llm=Ollama(model="llama3:8b")
response=llm.invoke("WRITE AN ARTICLE ON LLM & nlp")
print(response)