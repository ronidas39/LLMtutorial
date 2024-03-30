from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import io

loader=PyPDFLoader("Vedic_history.pdf")
docs=loader.load()
rs=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
text=""
for doc in docs:
    text=text+doc.page_content
chunks=rs.split_text(text)
llm=ChatOpenAI(model="gpt-4-0125-preview",temperature=0.0)
template="""
    You are an expert teacher at creating question and answer pair based on materials and documentation.
    Your goal is to prepare question & answer pair to help students for their study.
    create various types of questions as below:
    1.question with one line answer.
    2.question with few lines of answer.
    3.elaborative questions which requires more lines for answer
    You do this by using the text below:

    ------------
    {text}
    ------------

    
    Make sure not to lose any important information. Dont put sequence number on the question answer pair

    QUESTION:

    ANSWER:
    """

prompt=PromptTemplate.from_template(template)
with io.open("material.txt","w",encoding="utf-8")as f1:
    for chunk in chunks[:5]:
        msg=prompt.format(text=chunk)
        f1.write(llm.invoke(msg).content)
        f1.write("\n")
    f1.close()

