from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent,AgentType
from langchain.tools import Tool
from openai import OpenAI
import streamlit as st

client=OpenAI()


st.set_page_config(page_title="Design anything")
st.header="write anything you want to design"
input=st.text_input("enter your thoughts")

def genImage(input):
    response=client.images.generate(
        model="dall-e-3",
        prompt=input,
        size="1024x1024",
        quality="hd",
        n=1
    )
    url=response.data[0].url
    return url

llm=ChatOpenAI(model="gpt-4",temperature=0.0)
design=Tool(
    name="generateImage",
    func=genImage,
    description="use to generate image from generarteImage tool"
)
tools=[design]
agent=initialize_agent(tools,llm,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)
if st.button("Submit",type="primary"):
    if input is not None:
        response=agent.run(input)
        url="https://" + response.split("https://")[1].replace(")","")
        st.image(url,caption=input)

