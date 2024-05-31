
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain
llm=ChatOpenAI(model="gpt-4o")
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai")
search = GoogleSearchAPIWrapper()
web_research_retriever = WebResearchRetriever.from_llm(
vectorstore=vectorstore,
llm=llm,
search=search,
)
def gen_answer(user_input):
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=web_research_retriever) 
    result = qa_chain.invoke({"question": user_input})

    # we get the results for user query with both answer and source url that were used to generate answer
    return(result["answer"])

