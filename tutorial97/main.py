import streamlit as st
from pytube import YouTube
import os
from split_mp3 import split_audio
from transcript_generator import generate_transcript
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Initialize session state variables
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'input_url' not in st.session_state:
    st.session_state.input_url = ""
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

cwd = os.getcwd()

st.title("YouTube Video Chat App")

if st.session_state.step == 0:
    st.write("Enter your video URL:")
    st.session_state.input_url = st.text_area("Enter your video URL")
    if st.button("Submit"):
        if st.session_state.input_url:
            st.session_state.step = 1
            st.experimental_rerun()

if st.session_state.step == 1:
    st.video(st.session_state.input_url)
    yt = YouTube(st.session_state.input_url)
    data = yt.streams.filter(only_audio=True).first()
    output_file = data.download(output_path=cwd)
    os.rename(output_file, "sample.mp3")

    input_audio_path = os.path.join(cwd, "sample.mp3")
    output_folder = cwd
    segment_length = 600000  # Segment length for splitting audio

    split_audio(input_audio_path, output_folder, segment_length)
    st.success("All MP3s are split successfully")

    generate_transcript()
    st.success("Transcript file generated successfully")

    loader = TextLoader(os.path.join(cwd, "transcript.txt"))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=150)
    splits = text_splitter.split_text(docs[0].page_content)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(splits, embedding=embeddings, persist_directory=os.path.join(cwd, "output"))
    vs=Chroma(persist_directory=os.path.join(cwd, "output"),embedding_function=embeddings)
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4"),
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )

    st.session_state.step = 2
    st.experimental_rerun()

if st.session_state.step == 2:
    input_qsn = st.text_input("Ask your question")
    if st.button("Submit Question"):
        response = st.session_state.qa_chain.invoke(input_qsn)
        st.write(response["result"])
