from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage,AIMessage
import streamlit as st
import base64
from PIL import Image
chain=ChatOpenAI(model="gpt-4o",max_tokens=2048)
def encode_image(upload_file):
    image_bytes=upload_file.getvalue()
    base64_image=base64.b64encode(image_bytes).decode("utf-8")
    return base64_image

def get_response(b64image,qsn):
    msg=chain.invoke(
        [
            AIMessage(
            content="you are an useful asnd intelligent boty who is very good at image reading ocr taskto get insights from images of invoces"
            ),
            HumanMessage(
                content=[
                    {"type":"text","text":qsn},
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
    upload_file=st.file_uploader("upload your file",type=["jpg"])
    if upload_file is not None:
        image=Image.open(upload_file)
        st.image(image,caption="your invoice",use_column_width=True)
        st.success("image uploaded successfully")
        b64_image=encode_image(upload_file)
        qsn=st.text_area("ask your question")
        if qsn is not None:
            btn=st.button("submit")
            if btn:
                response=get_response(b64_image,qsn)
                st.write(response)

if __name__=="__main__":
    main()