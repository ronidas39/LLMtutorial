from langchain.document_loaders import DirectoryLoader
import os
import time

st = time.time()
pwd=os.getcwd()
loader = DirectoryLoader(pwd, glob="**/*.csv",use_multithreading=True)
docs=loader.load()
print(len(docs))
# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print(elapsed_time)