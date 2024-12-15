import chromadb
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

embed_model=OpenAIEmbedding()
loader=SimpleDirectoryReader(input_files=["/Users/roni/Documents/GitHub/LLMtutorial/tutorial126/ragdoc.txt"])
docs=loader.load_data()
db=chromadb.PersistentClient(path="./db")
cc=db.get_or_create_collection("tutorial126")
vs=ChromaVectorStore(chroma_collection=cc)
sc=StorageContext.from_defaults(vector_store=vs)
index=VectorStoreIndex.from_documents(docs,storage_context=sc,embed_model=embed_model)