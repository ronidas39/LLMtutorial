from langchain.document_loaders import PyPDFLoader
loader=PyPDFLoader(file_path="india.pdf",extract_images=True)
docs=loader.load_and_split()
print(docs[26])