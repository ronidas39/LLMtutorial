from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

urls=["https://www.espncricinfo.com/series/afghanistan-in-india-2023-24-1389384/india-vs-afghanistan-3rd-t20i-1389398/full-scorecard"]
loader=AsyncChromiumLoader(urls=urls)
htmldocs=loader.load()
tf=Html2TextTransformer()
fd=tf.transform_documents(htmldocs)
ts=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
splits=ts.split_documents(fd)
llm=ChatOpenAI(model="gpt-4",temperature=0.8)
embeddings=OpenAIEmbeddings()
chroma_db=Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)
query="Rinku Singh scores on the match between india vs afghanistan"
chain=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=chroma_db.as_retriever()
)
response=chain(query)
print(response)


