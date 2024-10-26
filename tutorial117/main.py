from langchain_community.document_loaders import RedditPostsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains.question_answering import load_qa_chain
llm=ChatOpenAI(model="gpt-4o")
loader=RedditPostsLoader(
                        client_id="7aJt9CuSBwg4unyMjk8kZg", 
                        client_secret="v1UUmCA4VHU-0MLzzR11pHYv7dRzXA", 
                        user_agent="Sad_Mud_4484", 
                        search_queries=["IndiaCricket"], 
                        mode= "subreddit", 
                        number_posts= 100
                         )
docs=loader.load()
ts=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
sd=ts.split_documents(docs)
vs=Chroma.from_documents(filter_complex_metadata(sd),embedding=OpenAIEmbeddings(),persist_directory="./vs")
vs=Chroma(persist_directory="./vs",embedding_function=OpenAIEmbeddings())
retriever=vs.as_retriever(k=5)
match_docs=retriever.invoke("india vs newzeland test")
print(match_docs)
chain=load_qa_chain(llm=llm,chain_type="stuff")
response=chain.run(input_documents=match_docs,question="how Washington Sundar performed?")
print(response)