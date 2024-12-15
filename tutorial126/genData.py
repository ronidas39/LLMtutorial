from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
import io,sys,time
with io.open("urls.csv","r",encoding="utf-8")as f1:
    data=f1.read()
    f1.close()

#llm=ChatOllama(model="llama3.2:latest")
llm=ChatOpenAI(model="gpt-4o")
template=""" for the given {text} extract every information related to {name} , dont miss anything related to {name} ,dont shorten the content try to use as it is"""
prompt=PromptTemplate.from_template(template)

lines=data.split("\n")
with io.open("ragdoc.txt","w",encoding="utf-8")as f1:
    for line in lines:
        url=line.split(",")[1]
        name=line.split(",")[0]
        loader=AsyncChromiumLoader([url],user_agent="MyAppUserAgent")
        htmldocs=loader.load()
        print(htmldocs)
        bs_transformer=BeautifulSoupTransformer()
        docs_transformed=bs_transformer.transform_documents(htmldocs,tags_to_extract=["div"])
        chain=prompt|llm
        response=chain.invoke({"name":name,"text":docs_transformed})
        print(name)
        #print(response.content)
        f1.write(str(response.content)+"\n\n")
        time.sleep(20)
    f1.close()