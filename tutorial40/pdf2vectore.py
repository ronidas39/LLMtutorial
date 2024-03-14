from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Qdrant
import os

def pdf2VectoreStore():
    loader = DirectoryLoader(path=os.getcwd()+"//output/", glob="./*.pdf", loader_cls=PyPDFLoader)
    docs=loader.load()
    full_text = ''
    for doc in docs:
        full_text += doc.page_content
    # Split the original text into lines
    lines = full_text.splitlines()
    non_empty_lines = []
    for line in lines:
        if line:
            non_empty_lines.append(line)
    full_text = " ".join(non_empty_lines)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc_chunks = text_splitter.create_documents([full_text])
    qdrant = Qdrant.from_documents(
    documents=doc_chunks,
    embedding=GPT4AllEmbeddings(),
    path="./tmp/local_qdrant",
    collection_name="arxiv_papers",
    )
    retriever = qdrant.as_retriever()
    return retriever
