
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage,AIMessage
import streamlit as st
import base64
from PIL import Image
from pymongo import MongoClient
import urllib,json,io
from response_gen import genresponse
chain=ChatOpenAI(model="gpt-4o",temperature=0.0)
#mongo client
username="ronidas"
pwd="t2HKvnxjL38QGV3D"
client=MongoClient("mongodb+srv://"+urllib.parse.quote(username)+":"+urllib.parse.quote(pwd)+"@cluster0.lymvb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db=client["invoice"]
collection=db["invoice"]

def encode_image(upload_file):
    image_bytes=upload_file.getvalue()
    base64_image=base64.b64encode(image_bytes).decode("utf-8")
    return base64_image

def get_response(b64image):
    msg=chain.invoke(
        [
            AIMessage(
            content="you are an useful asnd intelligent boty who is very good at image reading ocr taskto get insights from images of invoces"
            ),
            HumanMessage(
                content=[
                    {"type":"text","text":"""summarise the invoice into json with key value pair of the following keys: 
                    invoice_number
                    invoice_date
                    customer_name
                    product will be list of maps with brand,item,unit and single_unit_price,all_unit_price as keys
                    total_price 
                    mode_of_payment 
                    and return it as output
                    make sure to remove "$" from price related column and 
                    make sure values in all price related columns must be stored as number not a string
                    output will be only json nothing else this is very strict must follow"""},
                    {"type":"image_url",
                     "image_url":{
                         "url":"data:image/jpg;base64,"+ b64image,
                         "detail":"auto"
                                  }
                    
                    }
                ]
            )
        ]
    )
    return msg.content

def main():
    st.title("INVOICE ANALYSIS APP")
    upload_files=st.file_uploader("upload your file",type=["jpg"],accept_multiple_files=True)
    if upload_files is not None:
        for upload_file in upload_files:
            image=Image.open(upload_file)
            st.image(image,caption="your invoice",use_column_width=True)
            st.success("image uploaded successfully")
            b64_image=encode_image(upload_file)
            response=get_response(b64_image)
            data=response.replace("json","")
            data=data.replace("`","")
            data=json.loads(data)
            collection.insert_one(data)
        count=collection.count_documents({})
        if count >0:
            st.success("documents are uploaded successfully")
            qsn=st.text_area("ask your question")
            if qsn is not None:
                btn=st.button("submit")
                if btn:
                    response=genresponse(qsn)
                    for result in response:
                        st.write(result)


if __name__=="__main__":
    main()

