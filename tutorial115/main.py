from lumaai import LumaAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st
from genrate_video import generateVideo,extendVideo
import time
st.title("VIDEO GENERATOR APP")
duations=[5,10,15,20]
selectd_duration=st.selectbox("your option",duations)
input=st.text_area("enter your subject")
llm=ChatOpenAI(model="gpt-4o")
template="you are intelligent and creative agent who can expression imaginary thought about {input} in a single line with minimunm words"
prompt=PromptTemplate.from_template(template)
client=LumaAI()
chain=prompt|llm
if input is not None:
    btn=st.button("submit")
    if btn:
        response=chain.invoke({"input":input})
        video_prompt=response.content
        st.write(video_prompt)
        ids=[]
        n=int(selectd_duration/5)
        print(n)
        for i in range(int(selectd_duration/5)):
            if i==0:
                first_id=generateVideo(video_prompt)
                st.write(f"{i}th id is {first_id}")
                ids.append(first_id)
            else:
                next_id=extendVideo(video_prompt,ids[-1])
                st.write(f"{i}th id is {next_id}")
                ids.append(next_id)
        final_id=ids[-1]
        final_meta_data=client.generations.get(id=final_id)
        link=final_meta_data.assets.video
        st.write(link)
        time.sleep(10)
        st.video(data=link,autoplay=True)


