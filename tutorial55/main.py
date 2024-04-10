import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent,AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os,glob
cwd=os.getcwd()
old_files=glob.glob(cwd+"/*.png")
for file in old_files:
    os.remove(file)
llm=ChatOpenAI(model="gpt-4")
tools=[PythonREPLTool()]
prefix="""
            You are an expert data scientist who has very solid knowledge in csv operation & pandas dataframe 
            designed to analyse dataframe and whenever required use PythonREPLTool to analysis and 
            plot nice detailed charts, please remember the below for charts:
            all charts must be professional use attractive design and colors,
            always put exact numbers in chart to represent the values for each requested item,
            example for bar chart all bar should have the exact value displayed at the top of the bar .
            if only user ask then only create the chart else dont create ,all charts must be save as .png into this "/Users/roni/Documents/GitHub/LLMtutorial/tutorial55/" location.Take this a strong instruction.
            remember below points on the dataframe:
            dataframe has these columns :
            pizza_id 
            order_id
            pizza_name_id
            quantity
            order_date
            order_time
            unit_price
            total_price
            pizza_size
            pizza_category
            pizza_ingredients
            pizza_name

            use below description as well very strictly for creating charts:

            order_date gives the date of any order in mm-dd-yyyy format example 01-01-2015
            order_time gives the timing of any order in  hh:mm:ss format example 20:02:57 
            unit_price gives the order of single pizza
            total_price gives total price for specific order_id
            when monthwise is requested please convert the dates into month 
            when day is requested please convert the dates into day
            while plotting monthwise data month should start from january and go on from left to right
            while plotting daywise data day should start from monday and go on from left to right
            while plotting the chart always rotate the values in xaxis and axix labels to fit into the graph.
            please remember there are total 4 quaters , and these are as followed :
            1st querter =1st jan to 31st march
            2nd quarter= 1 april to 30th june
            3rd quarter =1st jule to 30th september
            4=1st october to 31st december


            if you cant get the result from dataframe, just return "I don't know" as the answer
            """

st.title("SALES REPORT ANALYSIS")
st.write("upload your csv here")
file=st.file_uploader("select your file",type=["csv"])
if file is not None:
    df=pd.read_csv(file)
    st.write(df)
    input=st.text_area("ask your question related to 9this sales report")
    if input is not None:
        button=st.button("Submit")
        if button:
            agent=create_pandas_dataframe_agent(
                llm,df,verbose=True,prefix=prefix,agent_type=AgentType.OPENAI_FUNCTIONS,extra_tools=tools
            )
            response=agent.invoke(input)
            st.write(response["output"])
            new_files=glob.glob(cwd+"/*.png")
            if new_files:
                st.image(new_files[0])

