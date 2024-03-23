import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Vectara
from langchain_community.document_loaders import S3DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
load_dotenv()
llm=ChatOpenAI(model="gpt-4-turbo-preview",temperature=0.0)
memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)

vs=Vectara(vectara_customer_id=os.getenv("VECTARA_CUSTOMER_ID"),
           vectara_corpus_id=os.getenv("VECTARA_CORPUS_ID"),
           vectara_api_key=os.getenv("VECTARA_API_KEY")
           )
loader=S3DirectoryLoader("llm-test-2024")
doc=loader.load()

vectara=Vectara.from_documents(doc,embedding=None)
retriever=vectara.as_retriever()
bot=ConversationalRetrievalChain.from_llm(llm,retriever,memory=memory,verbose=False)
result=bot.invoke({"What early civilizations in Greece were influential in shaping Western culture?"})
print(result["answer"])
