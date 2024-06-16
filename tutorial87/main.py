#imports
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage,AIMessage
import streamlit as st
import base64
from PIL import Image


#set our llm
chain=ChatOpenAI(model="gpt-4o",temperature=0.0)

#function to ecnode image with utf-8 string
def encode_image(upload_file):
    image_bytes=upload_file.getvalue()
    base64_image=base64.b64encode(image_bytes).decode("utf-8")
    return base64_image

# function to generate response
def get_response(b64image):
    msg=chain.invoke(
        [
            AIMessage(
            content="you are an useful and intelligent bot who is very good at image reading ocr task to get insights from images of cars"
            ),
            HumanMessage(
                content=[
                    {"type":"text","text":"""read the car image and check if the number plate has number or not ,
                    if no number is there respond "invalid car" else respond "valid car"
                    output will be only either of above two only, nothing else you must follow this"""},
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


#main function with streamlit ui and function call
def main():
    st.title("CAR ANALYSIS APP")
    upload_files=st.file_uploader("upload your file",type=["jpg"],accept_multiple_files=True)
    if upload_files is not None:
        for upload_file in upload_files:
            image=Image.open(upload_file)
            st.image(image,caption="your car",use_column_width=True)
            st.success("image uploaded successfully")
            b64_image=encode_image(upload_file)
            response=get_response(b64_image)
            st.write(response)
            
if __name__=="__main__":
    main()
