from streamlit_mic_recorder import mic_recorder,speech_to_text
from langchain_openai import ChatOpenAI
import streamlit as st
llm=ChatOpenAI(model="gpt-4o")
st.title("Your One Stop Voice Assistant")
st.write("Voice enabled Chat App")

# audio=mic_recorder(start_prompt="**",stop_prompt="##",key="recorder")
# if audio:
#     st.audio(audio["bytes"])

text=speech_to_text(language="en",use_container_width=True,just_once=True,key="STT")
if text:
    response=llm.invoke(text)
    st.write(response.content)
