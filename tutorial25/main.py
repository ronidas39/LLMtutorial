from langchain.document_loaders import UnstructuredHTMLLoader,BSHTMLLoader,DirectoryLoader
import os

#loader=UnstructuredHTMLLoader(file_path="sample.html")
#loader=BSHTMLLoader(file_path="sample.html")
loader=DirectoryLoader(path=os.getcwd(),glob="**/*.html",loader_cls=BSHTMLLoader)
docs=loader.load()
print(docs)