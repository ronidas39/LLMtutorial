import pandas as pd
from pandasai import SmartDataframe
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(model="gpt-4o")

df=pd.read_csv(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial98\output.csv")
sdf=SmartDataframe(df,config={"llm":llm})
response=sdf.chat("who played longest duration in years?")
print(response)