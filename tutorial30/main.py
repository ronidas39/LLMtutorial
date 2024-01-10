from langchain.document_loaders import youtube
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import streamlit as st


st.set_page_config(page_title="YOUTUBE VIDEO SUMMARISER APP")
st.header="YOUR YOUTUBE VIDEO URL"
url=st.text_input("enter the video url from youtube")

if st.button("Submit",type="primary"):
    if url is not None:
        print(url)
        loader=youtube.YoutubeLoader.from_youtube_url(url)
        docs=loader.load()
        ts=RecursiveCharacterTextSplitter(chunk_size=7000,chunk_overlap=0)
        fd=ts.split_documents(docs)
        l=[]
        for xx in fd:
            response=openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role":"system","content":"you are a helpful intelligent AI assistant"},
                    {"role":"user","content":f"summarize the following paragraph into bullet points:\n\n{xx}"}

                ]
            )
            msg=response["choices"][0]["message"]["content"]
            l.append(msg)
        st.write("".join(l))

           
