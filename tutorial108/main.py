from streamlit_mic_recorder import speech_to_text
import streamlit as st
from emailAgent import runAgent

st.title("your one stop voice enable email Assistant")
st.write("instruct your voice and see the magic")
text=speech_to_text(language="en",use_container_width=True,just_once=True,key="STT")
if text:
    response=runAgent(text)
    st.write(response)