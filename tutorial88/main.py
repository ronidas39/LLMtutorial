import streamlit as st
from PIL import Image
import base64
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage,AIMessage

llm=ChatOpenAI(model="gpt-4o")


def encode_image(upload_file):
    image_bytes=upload_file.getvalue()
    base64_image=base64.b64encode(image_bytes).decode("utf-8")
    return base64_image


def gen_response(base64_image,input):
    response=llm.invoke(
        [
            AIMessage(
                content="you are an useful & intelligent bot who is very good at image reading related ocr tassks to get insights from images uploaded by user"
            ),
            HumanMessage(
                content=[
                    {"type":"text","text":input},
                    {"type":"image_url",
                     "image_url":{
                         "url":"data:image/jpg;base64,"+base64_image,
                         "detail":"auto"
                     }
                     }
                ]
            )
        ]

    )
    return response.content

def main():
    st.title("OBJECT DETECTION & ANALYSIS APP")
    upload_file=st.file_uploader("upload your image here",type=["jpg"])
    if upload_file is not None:
        image=Image.open(upload_file)
        st.image(image,caption="your image",use_column_width=True)
        st.success("image uploaded successfully")
        base64_image=encode_image(upload_file)
        input=st.text_area("ask your question here")
        btn=st.button("submit")
        if btn:
            response=gen_response(base64_image,input)
            st.write(response)

        

if __name__=="__main__":
    main()