import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from genvideo import genvideo
from downloadvideo import download_video
import time
llm=ChatOpenAI(model="gpt-4",temperature=0.0)
ts="""
you are a news anchor for a global news channel,with this context genereate a concise summary of ther following
{text}
"""
pt=PromptTemplate(template=ts,input_variables=["text"])
st.set_page_config(page_title="24/7 NEWS CHANNEL POWERED BY AI DRIVEN NEWS ANCHOR")
st.header="what you want to hear and watch"
qsn=st.text_area("enter your query")
search=DuckDuckGoSearchResults(backend="news")
if st.button("Submit",type="primary"):
    if qsn is not None:
        result=search.run(qsn)
        data=result.replace("[snippet: ","")
        data=data[:-1]
        docs=[Document(page_content=t) for t in data]
        chain=load_summarize_chain(llm,chain_type="stuff",prompt=pt)
        summary=chain.run(docs)
        id=genvideo("https://clips-presenters.d-id.com/lana/uXbrIxQFjr/kzlKYBZ2wc/image.png",summary,"en-US-JaneNeural")
        time.sleep(100)
        url=download_video(id)
        st.video(url)


