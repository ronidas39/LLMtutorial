
#for ui and accepting user input as audio , play output audio file
import streamlit as st
from streamlit_mic_recorder import speech_to_text

#parsing and extracting cituy
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain_community.utilities import OpenWeatherMapAPIWrapper
import json
from audio_gen import generate_audio

os.environ["OPENWEATHERMAP_API_KEY"] = "90cd0252d44c2c0223aa6bff746d9626"
weather = OpenWeatherMapAPIWrapper()


llm=ChatOpenAI(model="gpt-4o")
prompt="""
you are an intelligent assistant who is very good in world geography , your main task is to extract the city name from a sentence{instruction} 
and return output as json , while returning the json use the below key :
city
please note you have to only return the json nothing else
"""
def genresponse(input):
    query_with_prompt=PromptTemplate(
        template=prompt,
        input_variables=["instruction"]
    )
    llmchain=LLMChain(llm=llm,prompt=query_with_prompt,verbose=True)
    response=llmchain.invoke(
        {"instruction":input}
                             )
    data=response["text"]
    data=data.replace("json","")
    data=data.replace("`","")
    data=json.loads(data)
    return data

llm=ChatOpenAI(model="gpt-4o")
st.title("VOICE ENABLED NEWS APP")
st.write("ask anything")
text=speech_to_text(language="en",use_container_width=True,just_once=True,key="STT")
if text is not None:
    data=genresponse(text)
    city_name=data["city"]
    st.write(city_name)
    weather_data = weather.run(city_name)
    st.write(weather_data)
    generate_audio(weather_data)
    st.audio(data="output.wav",format="audio/wav",autoplay=True)