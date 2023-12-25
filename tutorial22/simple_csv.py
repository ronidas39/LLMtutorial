from langchain.document_loaders.csv_loader import CSVLoader
loader=CSVLoader("sample.csv")
doc=loader.load()
for d in doc:
    print(d.page_content)