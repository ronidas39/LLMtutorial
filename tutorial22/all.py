from langchain.document_loaders import DirectoryLoader
import os
loader=DirectoryLoader(path=os.getcwd(),glob="**/*.csv")
docs=loader.load()
print(len(docs))