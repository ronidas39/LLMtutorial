
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

template="""
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use five sentences minimum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
loader=TextLoader(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial76\test.txt")
documents=loader.load()
text_splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=0)
docs=text_splitter.split_documents(documents)
embeddings=OpenAIEmbeddings()

# vector_db=ElasticsearchStore.from_documents(
#     docs,
#     embedding=embeddings,
#     index_name="tutorial76",
#     es_cloud_id="tt:dXMtZWFzdC0yLmF3cy5lbGFzdGljLWNsb3VkLmNvbTo0NDMkZjdmOWIwMTBjNTRhNDg4NDk2MTY4OGE0NjM5NTkwZmEkMmQ0MTQ3NDU0NjEwNGY3OWE4Y2E4MWEyNDAwYTQwZDQ=",
#     es_api_key="Vnl0SjRZOEJkQ2RfOUxjNndRZ1Q6RHo3bHF5RVVTZGVHajZMU3FTd1lOUQ=="
# )

vector_db=ElasticsearchStore(
    embedding=embeddings,
    index_name="tutorial76",
    es_cloud_id="tt:dXMtZWFzdC0yLmF3cy5lbGFzdGljLWNsb3VkLmNvbTo0NDMkZjdmOWIwMTBjNTRhNDg4NDk2MTY4OGE0NjM5NTkwZmEkMmQ0MTQ3NDU0NjEwNGY3OWE4Y2E4MWEyNDAwYTQwZDQ=",
    es_api_key="Vnl0SjRZOEJkQ2RfOUxjNndRZ1Q6RHo3bHF5RVVTZGVHajZMU3FTd1lOUQ=="
)
retriever=vector_db.as_retriever()
prompt=ChatPromptTemplate.from_template(template)
llm=ChatOpenAI(model="gpt-4o",temperature=0)
rag_chain=(
    {"context":retriever,"question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
query="given some family information on king john"
response=rag_chain.invoke(query)
print(response)