import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import io,glob,base64,sys
from langchain.schema.document import Document
import uuid
from langchain_community.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from PIL import Image 
from langchain.schema.messages import HumanMessage
import streamlit as st
llm=ChatOpenAI(model="gpt-4o")
# Create chroma
vectorstore = Chroma(
    collection_name="mm_rag_clip_photos", persist_directory=os.getcwd()+"/index",embedding_function=OpenCLIPEmbeddings()
)

# def image2base64(ip):
#     with Image.open(ip) as image:
#         buffer=io.BytesIO()
#         image.save(buffer,format=image.format)
#         img_str=base64.b64encode(buffer.getvalue())
#         return img_str.decode("utf-8")
    
# cwd=os.getcwd()
# files=glob.glob(cwd+"/*.png")
# imgs=[]
# imgs_summary=[]

# for file in files:
#     image_str=image2base64(file)
#     response=llm.invoke(
#         [
#             HumanMessage(
#                 content=[
#                     {"type":"text","text":"please give a summary of the image provided , be descriptive and smart"},
#                     {"type":"image_url","image_url":
#                      {
#                         "url":f"data:image/png;base64,{image_str}"

#                     },
#                     },
#                 ]
#             )
#         ]
#     )
#     imgs.append(image_str)
#     imgs_summary.append(response.content)
#     print(imgs_summary)
# documents=[]
# for e,s in zip(imgs,imgs_summary):
#         i=str(uuid.uuid4())
#         doc=Document(
#             page_content=s,
#             metadata={
#                 "id":i,
#                 "type":"image",
#                 "original_content":e
#             }
#         )
#         documents.append(doc)
# vectorstore.add_documents(documents=documents)
retriver=vectorstore.as_retriever()
chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriver
    )
# response=chain.invoke("describe all the characters you found here")
# print(response)

response1=vectorstore.similarity_search("describe  fight scenes",k=3)
context=""
ri=[]
for xx in response1:
   if xx.metadata["type"] =="image":
      context +='[image]'+xx.page_content
      ri.append(xx.metadata["original_content"])

print(context)
image_output=ri[0].encode("utf-8")
st.image(base64.b64decode(image_output))