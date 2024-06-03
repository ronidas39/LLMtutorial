from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
loader=TextLoader(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial75\test.txt")
documents=loader.load()
text_splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=0)
docs=text_splitter.split_documents(documents)
embedding=OpenAIEmbeddings()
elastic_vector_store=ElasticsearchStore.from_documents(
    docs,
    index_name="tutorial75",
    es_api_key="X3hBXzNvOEI4NjQ3aTdxQjV1UGo6OHRNYzFFSUZRRi12THFkOUFCQkQxdw==",
    embedding=embedding,
    es_cloud_id="814947a9330a4e89941b9478109934f4:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyRjMTg5ZDVhNGFiMzE0ZDA2OWFkOTcwM2QxNTQ0NjU3YiRkZTBhZTY3OTI2MWI0ZjdiYjYyODQ2ODk0Y2U2YWNjZA=="

)
print(elastic_vector_store)