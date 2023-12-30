from langchain.document_loaders import DirectoryLoader
import os , time
pwd=os.getcwd()
st=time.time()

loader=DirectoryLoader(path=pwd,glob="**/*.csv",use_multithreading=True)
docs=loader.load()
endt=time.time()
et=endt-st
print(et)
print(len(docs))