from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

llm=ChatOpenAI(model="gpt-4-turbo-preview",temperature=0.0,max_tokens=1024)
template="""
analyse the below set of comments and generate public sentiment on BJP for the upcoming election 2024,
just write one sentences as "possitive for BJP" if you find comments are supporting bjp ,
else just write "negative for BJP" if you find comments are not supporting BJP
{context}
"""
prompt=ChatPromptTemplate.from_template(template)

loader=TextLoader("comment.txt")
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=7000,chunk_overlap=0)
texts=text_splitter.split_documents(docs)

for text in texts:
    msg=prompt.format(context=text)
    print(llm.invoke(msg).content)