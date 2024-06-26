import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_openai import ChatOpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
import os,glob
pwd=os.getcwd()

llm=ChatOpenAI(model="gpt-4o")
st.title("CHAT WITH YOUR EXCEL FILE")
file=st.file_uploader("upload your file",type=["xlsx"])
if file:
    df=pd.read_excel(file)
    sdf=SmartDataframe(df,
                       config={"llm":llm,"response_parser":StreamlitResponse,"save_charts":True,
                               "save_charts_path":pwd
                               }
    )
    options=["chat","plot"]
    selected_option=st.selectbox("choose an option",options)
    if (selected_option=="chat"):
        input=st.text_area("ask your question here")
        if input is not None:
            btn=st.button("submit")
            if btn:
                response=sdf.chat(input)
                st.write(response)
    if (selected_option=="plot"):
        file=glob.glob(pwd+"/*.png")
        if file:
            os.remove(file[0])
        input=st.text_area("ask your question here")
        if input is not None:
            btn=st.button("submit")
            if btn:
                response=sdf.chat(input)
                file=glob.glob(pwd+"/*.png")
                if file:
                    st.image(image=file[0],caption="plot for :"+ input,width=1024)
            
                


