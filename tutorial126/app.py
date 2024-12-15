import chromadb
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from openai import OpenAI
import streamlit as st 
client=OpenAI()
st.title("AYURDEVA WITH AI")
query=st.text_input("ask your question")
db=chromadb.PersistentClient(path="./db")
cc=db.get_or_create_collection("tutorial126")
vs=ChromaVectorStore(chroma_collection=cc)
index=VectorStoreIndex.from_vector_store(vector_store=vs)
if query is not None:
    btn=st.button("submit")
    if btn:
        retriver=index.as_retriever(similarity_top_k=5)
        nodes=retriver.retrieve(query)
        docs=""
        for node in nodes:
            docs=docs+node.text
        #st.write(docs)
        system_prompt=f"""you are an intelligent ai assistant who has expertise in ayurvedic can write nice and well crafted articles ,use only {docs} as context to write answer the question asked by user, dont use anything else other than this{docs},make it professional writing,detailed use headers and bullets"""
        response=client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":query}
            ]
        )
        st.write(response.choices[0].message.content)
