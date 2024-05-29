import streamlit as st
from streamlit_mic_recorder import speech_to_text
from langchain_openai import ChatOpenAI
from audio_gen import generate_audio

llm=ChatOpenAI(model="gpt-4o")
st.title("VOICE ENABLED CHAT APP")
st.write("ask anything")
text=speech_to_text(language="en",use_container_width=True,just_once=True,key="STT")
if text:
    response=llm.invoke(text)
    content=response.content
    st.write(content)
    generate_audio(content)
    st.audio(data="output.wav",format="audio/wav",autoplay=True)