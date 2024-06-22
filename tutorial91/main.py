from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json,base64
import streamlit as st
from fetch_price import getPrice
from genrate_table import gentable


llm=ChatOpenAI(model="gpt-4o",temperature=0)
prompt="""
you are an intelligent bot who can translate any {instruction} into a json document with the following keys:
product_name
quantity_volume

output will be only json nothing else
"""

def show_pdf(file_path):
    with open(file_path,"rb") as f1:
        base64_pdf=base64.b64encode(f1.read()).decode("utf-8")
    pdf_display=f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1200" height="900" type="application/pdf"></iframe>'
    st.markdown(pdf_display,unsafe_allow_html=True)

def genresponse(input):
    query_with_prompt=PromptTemplate(
        template=prompt,
        input_variables=["instruction"]
    )
    llmchain=LLMChain(llm=llm,prompt=query_with_prompt)
    response=llmchain.invoke({"instruction":input})
    data=response["text"]
    data=data.replace("json","")
    data=data.replace("`","")
    data=json.loads(data)
    return data

st.title("INVOICE GENERATOR APP")
st.write("what you want to order")
input=st.text_area("write your requirements")
if input is not None:
    btn=st.button("submit")
    if btn:
        response=genresponse(input)
        if isinstance(response,dict):
            name=response["product_name"]
            unit=response["quantity_volume"][0]
            price=getPrice(name,unit)
            response["price"]=price
        if isinstance(response,list):
            for item in response:
                name=item["product_name"]
                unit=item["quantity_volume"][0]
                price=getPrice(name,unit)
                item["price"]=price
        final_data=[["NAME","QUANTITY_VOLUME","PRICE"]]
        for data in response:
            product_lists=[data["product_name"],data["quantity_volume"],data["price"]]
            final_data.append(product_lists)
        st.write(final_data)
        gentable(final_data)
        file_path=r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial91\invoice.pdf"
        show_pdf(file_path)

        
