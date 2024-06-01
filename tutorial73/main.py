import streamlit as st
from langchain_openai import ChatOpenAI
import base64
from PIL import Image
from langchain.schema.messages import HumanMessage,AIMessage
llm=ChatOpenAI(model="gpt-4o",max_tokens=2048)

def encode_image(upload_file):
    image_bytes=upload_file.getvalue()
    base64_image=base64.b64encode(image_bytes).decode("utf-8")
    return base64_image


def get_response(b64image):
    msg=llm.invoke(
        [
            AIMessage(
                content="""
you are intelligent assistant who can solve any mathematical problems on derivatives, 
you will be shared with a image where each line contains a problem ,
your task will be to solve all of them with possible explanation and step by step solutions
and formulas , then create a complete answer book
"""
            ),
            HumanMessage(
                content=[
                    {"type":"image_url",
                     "image_url":{
                         "url":"data:image/jpg;base64, " + b64image,
                         "detail":"auto"
                     }
                    }
                ]
            )
        ]
    )
    content=msg.content
    content=content.replace("\\","")
    return content




def main():
    st.title="Student Assignment Solver"
    upload_file=st.file_uploader("upload your assignment:",type=["jpg","png"])
    if upload_file is not None:
        image=Image.open(upload_file)
        st.image(image,caption="your assignment",use_column_width=True)
        st.text("your assignment uploaded successfully")
        base64_image=encode_image(upload_file)
        btn=st.button("submit")
        if btn:
            response=get_response(base64_image)
            print(response)
            st.markdown(response,unsafe_allow_html=True)

    
if __name__=="__main__":
    main()  
       
            

