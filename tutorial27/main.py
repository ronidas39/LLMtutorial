from langchain.document_loaders import PyPDFLoader
loader=PyPDFLoader(file_path="india.pdf")
docs=loader.load_and_split()
#for doc in docs:
    #print(doc)
    #print("========================================")

for i in range(len(docs)):
    print(docs[i].page_content+"\n\n")