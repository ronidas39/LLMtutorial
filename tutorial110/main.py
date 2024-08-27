from streamlit_mic_recorder import speech_to_text
import streamlit as st
from emailagent import runagent
from reademail import readbody
from adduser import createuser

st.title("voice driven service desk agent")
st.write("ask anything")
text=speech_to_text(language="en",use_container_width=True,just_once=True,key="STT")
if text:
    output=runagent(text)
    if output:
        response=readbody(output)
        userdata=createuser(response["username"])
        st.write(userdata)
        



