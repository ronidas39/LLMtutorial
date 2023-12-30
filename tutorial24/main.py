from langchain.document_loaders import DirectoryLoader
import os

pwd=os.getcwd()

loader=DirectoryLoader(path=pwd,glob="**/*.*")
docs=loader.load()

for doc in docs:
    #print(doc.metadata["source"])
    print(doc.page_content)
    print("==============================")