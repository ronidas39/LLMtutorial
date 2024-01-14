from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import openai,sys
from langchain.text_splitter import RecursiveCharacterTextSplitter

url="https://python.langchain.com/docs/use_cases/question_answering/"
loader=AsyncChromiumLoader([url])
tt=Html2TextTransformer()
docs=tt.transform_documents(loader.load())
ts=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=0)
fd=ts.split_documents(docs)
print(len(fd))
l=[]
for xx in fd:
    response=openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"system","content":"you are a helpful intelligent assistant"},
                  {"role":"user","content":f"summarize the following into bullet points,only consider meaningfull sentences, also ignbore all headings and words:\n\n{xx}"}
        ]
    )
    l.append(response["choices"][0]["message"]["content"])
print("".join(l))

