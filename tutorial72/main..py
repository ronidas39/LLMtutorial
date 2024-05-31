import streamlit as st
from streamlit_mic_recorder import speech_to_text
from langchain_openai import ChatOpenAI
from responsegen import gen_answer
from audio_gen import generate_audio

llm=ChatOpenAI(model="gpt-4o")
st.title("VOICE ENABLED WEB RESEARCH APP")
st.write("ask anything")
text=speech_to_text(language="en",use_container_width=True,just_once=True,key="STT")
if text:
    response=gen_answer(text)
    generate_audio(response)
    st.audio(data="output.wav",format="audio/wav",autoplay=True)