import streamlit as st
import base64
from langchain.schema.messages import HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from PIL import Image
chain=ChatOpenAI(model="gpt-4O",max_tokens=1024)

def encode_image(upload_file):
    image_bytes=upload_file.getvalue()
    base64_image=base64.b64encode(image_bytes).decode("utf-8")
    return base64_image

def get_response(b64image,qsn):
    msg=chain.invoke(
        [
            AIMessage(
            content="you are a useful and intelligent bot who is very good at ocr related task , such getting insights from images of invoices"
        ),
        HumanMessage(
            content=[
                {"type":"text","text":qsn},
                {
                   "type": "image_url",
                   "image_url":f"data:image/jpg;base64,{b64image}"
                }
            ]
        )
        ]
    )
    return msg.content


# if "conversation" not in st.session_state:
#     st.session_state.conversation=[]



def main():
    st.title("INVOICE ANALYSIS SYSTEM")
    upload_file=st.file_uploader("upload the invoice image..",type=["jpg"])
    if upload_file is not None:
        image=Image.open(upload_file)
        st.image(image,caption="your uploaded invoice",use_column_width=True)
        st.write("invoice image uploaded successfully")
        b64_image=encode_image(upload_file)
        st.success("image converted successfully")
        user_question=st.text_input("ask anything related to the invoice")
        submit_button=st.button("Submit")
        if submit_button and user_question:
            response=get_response(b64_image,user_question)
            st.write(response)




if __name__=="__main__":
    main()