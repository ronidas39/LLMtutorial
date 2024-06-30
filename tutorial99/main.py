import streamlit as st
from pandasai import SmartDataframe
from langchain_openai import ChatOpenAI
import pandas as pd

llm=ChatOpenAI(model="gpt-4o")
student_df=pd.read_csv(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial99\students.csv")
student_sdf=SmartDataframe(student_df,config={"llm":llm})
bmw_df=pd.read_csv(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial99\bmw.csv")
bmw_sdf=SmartDataframe(bmw_df,config={"llm":llm})

st.title("CHAT APP WITH MULTIPLE CSV")
options=["bmw","student"]
user_selection=st.selectbox("select your file",options)

if user_selection=="bmw":
    input1=st.text_input("ask your question")
    if input1 is not None:
        btn1=st.button("submit")
        if btn1:
            response=bmw_sdf.chat(input1)
            st.write(response)
elif user_selection=="student":
    input2=st.text_input("ask your question")
    if input2 is not None:
        btn2=st.button("submit")
        if btn2:
            response=student_sdf.chat(input2)
            st.write(response)
