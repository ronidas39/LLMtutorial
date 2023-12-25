from langchain.document_loaders import TextLoader

loader=TextLoader("simple.txt")
doc=loader.load()
for d in doc:
    print(d.page_content)