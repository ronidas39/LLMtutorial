from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup  
import io

url="https://docs.llamaindex.ai/en/stable/"
loader=RecursiveUrlLoader(url=url,
             max_depth=1000,
             extractor=lambda x:Soup(x,"html.parser").text
             )
docs=loader.load()
print(docs)
with io.open("url.txt","w",encoding="utf-8")as f1:
    for doc in docs:
        f1.write(doc.metadata["source"]+"\n")
    f1.close()