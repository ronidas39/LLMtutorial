import pandas as pd
df=pd.read_csv(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial91\data.csv")
df["price"]=pd.to_numeric(df["price"],errors="coerce")

def getPrice(name,unit):
    dfp=df[df["product"].str.lower()==name.lower()]
    price=dfp["price"].values[0]*float(unit)
    return price

