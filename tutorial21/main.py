from langchain.document_loaders import UnstructuredExcelLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
loader=UnstructuredExcelLoader("trending_football_players.xlsx")
index=VectorstoreIndexCreator()
doc=index.from_loaders([loader])
chain=RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0,model="gpt-4"),chain_type="stuff",retriever=doc.vectorstore.as_retriever(),input_key="question")
query="who has the lowest total stats"
response=chain({"question":query})
print(response)