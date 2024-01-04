from langchain.text_splitter import RecursiveCharacterTextSplitter
with open("sample.txt") as f1:
    data=f1.read()
    f1.close()

ts=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20,length_function=len,is_separator_regex=False)
texts=ts.create_documents([data])
for text in texts:
    print(text.page_content+"\n\n")