import pandas as pd
df = pd.read_csv("zip.csv")
def get_zip(city):
    x = df[df['city'] == city]
    zip=x["zip"].values[0]
    return zip

def get_state(code):
    x = df[df['state_id'] == code]
    state=x["state_name"].values[0]
    return state

