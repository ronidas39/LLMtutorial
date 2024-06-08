from typing import Union
import pandas as pd
from fastapi import FastAPI

df=pd.read_csv(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial80\uszips.csv")
app = FastAPI()


@app.get("/getzip/{name}")
def getZip(name):
    df1=df[df["city"]==name]
    data=str(df1["zip"].values[0])
    return {"Zipcode": "00"+data}


