import arxiv,os,glob
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.schema.runnable import RunnableParallel,RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

path="C:\\Users\\welcome\\OneDrive\\Documents\\GitHub\\LLMtutorial\\tutorial41\\output\\"



if "last_selected_option" not in st.session_state:
    st.session_state["last_selected_option"]=None
if "docs_processed" not in st.session_state:
    st.session_state["docs_processed"]=False
if "retriever" not in st.session_state:
    st.session_state["retriever"]=None


llm=ChatOpenAI(model="gpt-4",temperature=0.0,max_tokens=1024)
template="""
Answer the question based only on the following context: 
{context}
Question:{question}
"""
prompt=ChatPromptTemplate.from_template(template)

def download_parse(selected_option):
    client=arxiv.Client()
    search=arxiv.Search(
        query=selected_option,
        max_results=15,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results=client.results(search)
    for result in results:
        try:
            result.download_pdf(dirpath=path)
        except Exception as e:
            print(result)


def setoutput(input_text,retriever):
    chain=(RunnableParallel({"context":retriever,"question":RunnablePassthrough()})
          | prompt
          | llm
          | StrOutputParser()
    )
    result=chain.invoke(input_text)
    return result




st.title("Multi Specialty Research Assistant")
col1,col2=st.columns(2)
with col1:
    st.header("SELECT YOUR DOMAIN FOR RESEARCH")
    options=["healthcare","mathematics","physics","chemistry","AI","computer science","space research","quantum computing"]
    selected_option=st.selectbox("choose your domain for research",options,index=0,key="select_option")
    if selected_option:
        if selected_option != st.session_state["last_selected_option"]:
            st.session_state["docs_processed"]=False
            st.session_state["last_selected_option"]=selected_option
        if selected_option and not st.session_state["docs_processed"]:
            files=glob.glob(path+"*.*")
            for file in files:
                os.remove(file)
            download_parse(selected_option)
            loader=DirectoryLoader(path=path,glob="./*.pdf",loader_cls=PyPDFLoader)
            docs=[]
            try:
                docs=loader.load()
            except Exception as e:
                print(f"error load docs{e}")
            full_text=""
            for doc in docs:
                full_text +=doc.page_content
                lines=full_text.splitlines()
                non_empty_lines=[]
            for line in lines:
                if line:
                    non_empty_lines.append(line)
            full_text="".join(non_empty_lines)
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=50)
            doc_chunks=text_splitter.create_documents([full_text])
            vs=FAISS.from_documents(documents=doc_chunks,embedding=OpenAIEmbeddings())
            retriever=vs.as_retriever()
            vs.save_local("research_index")
            st.session_state["docs_processed"]=True
            st.session_state["retriever"]=retriever
            st.success("Documents are processed and stored into vector db")
        input_text=st.text_area("User Question Section",f"ask question related to topic {selected_option}",key="input_text")
if st.button("Submit",type="primary"):
    if st.session_state["retriever"] is not None:
        result=setoutput(input_text,st.session_state["retriever"])
    with col2:
        st.header("OUTPUT SECTION")
        st.write(result)
else:
    with col2:
        st.header("OUTPUT SECTION")
        st.write("your output will be generated by AI once you hit the submit button")
 

