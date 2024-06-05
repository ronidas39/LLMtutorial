from fastapi import FastAPI
from get_info import get_zip,get_state
app = FastAPI()



@app.get("/get/zip_code/{city}")
def read_zip(city):
    zip=get_zip(city)
    return {"zip":str(zip)}



@app.get("/get/state_name/{id}")
def read_zip(id):
    name=get_state(id)
    return {"name":name}
