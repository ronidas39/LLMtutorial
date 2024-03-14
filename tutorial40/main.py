import streamlit as st
import arxiv,os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import ChatPromptTemplate
from pdf2vectore import pdf2VectoreStore
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
qa_chain=LLMChain(
    llm=ChatOpenAI(model="gpt-4",max_tokens=1024),
    prompt=prompt
)
# Function to handle the logic after submission
def handle_submission(selected_option, input_text):
    # Example logic to generate output based on the input text
    # This can be customized based on your needs
    output_text = f"You selected '{selected_option}' and submitted: '{input_text}'"
    return output_text

def download_docs(selected_option):
    client = arxiv.Client()

# Search for the 10 most recent articles matching the keyword "quantum."
    search = arxiv.Search(
    query = "quantum",
    max_results = 10,
    sort_by = arxiv.SortCriterion.SubmittedDate
    )

    results = client.results(search)
    for result in results:
        result.download_pdf(dirpath=os.getcwd()+"\\output\\")






# Setting up the layout
st.title("Example Streamlit UI")

# Creating two columns for the left and right sides
col1, col2 = st.columns(2)

# Working on the left side (Column 1)
with col1:
    st.header("Input Section")
    # Dropdown menu for selecting an option
    options = ["healthcare", "mathematics", "physics","environment","computer science","AI","Quantum computing","space research"]
    selected_option = st.selectbox("Choose an option", options, index=0, key="select_option")

    # Activating text box and submit button based on dropdown selection
    if selected_option:
        download_docs(selected_option)
        context=pdf2VectoreStore()
        print("vector store created)")
        input_text = st.text_area("Text Box", f"Text for {selected_option}", key="input_text")
        submit_button = st.button("Submit")
        
# Handling submission and displaying output on the right side (Column 2)
if submit_button:
    output = qa_chain.run({"context":context,"question":input_text})

    with col2:
        st.header("Output Section")
        st.write(output)
else:
    # Showing a message or placeholder in the output section before submission
    with col2:
        st.header("Output Section")
        st.write("Your output will appear here after submission.")
