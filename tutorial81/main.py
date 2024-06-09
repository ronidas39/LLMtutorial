from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader,TextLoader
from langchain_community.document_loaders.merge import MergedDataLoader
loader1=WebBaseLoader("https://www.nature.com/natmachintell/?gad_source=1&gclid=CjwKCAjwgpCzBhBhEiwAOSQWQfzJ7tpFVRMJ7wWBhwNxSZ0tvsEUlAjDujKvg6dtdniReQTS-ZKQ-hoCTBYQAvD_BwE")
loader2=PyPDFLoader(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial81\sample.pdf")
loader3=TextLoader(r"C:\Users\welcome\OneDrive\Documents\GitHub\LLMtutorial\tutorial81\sample.txt")
loader=MergedDataLoader([loader1,loader2,loader3])
docs=loader.load()
print(docs)



