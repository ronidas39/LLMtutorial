import streamlit as st
import os
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter,PdfFormatOption
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM,OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
docs=[]
llm=OllamaLLM(model="gemma3:27b")
embedding=OllamaEmbeddings(model="llama3.2:3b")
IMAGE_RESOLUTION_SCALE=2.0
st.title("build your rag store")
uf=st.file_uploader("upload your file",type=["pdf"],accept_multiple_files=False)
if uf:
    file_name=uf.name
    with open(file_name,"wb")as f1:
        f1.write(uf.getbuffer())
        f1.close()
    st.success("uploaded successfully")
    input_doc_path=Path("/Users/ronidas/Documents/Tutorial129/sample_data.pdf")
    st.write(str(input_doc_path))
    output_dir=Path("images")
    pipeline_options=PdfPipelineOptions()
    pipeline_options.images_scale=IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images=True
    pipeline_options.generate_picture_images=True
    doc_converter=DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                        }
                        )
    conv_res=doc_converter.convert(input_doc_path)
    data=conv_res.document.export_to_dict()
    text=conv_res.document.export_to_text()
    text_doc=Document(
        page_content=text,
        metadata={"source":"text-content"}
    )
    docs.append(text_doc)
    pics=data["pictures"]
    page_nums=[]
    for pic in pics:
        page_nums.append(pic["prov"][0]["page_no"])

    page_nums=sorted(list(set(page_nums)))
    output_dir.mkdir(parents=True,exist_ok=True)
    doc_filename=conv_res.input.file.stem
    for page_no,page in conv_res.document.pages.items():
        page_no=page.page_no
        if page_no in page_nums:
            print(f"{page_no} is being processed")
            filename="/Users/ronidas/Documents/Tutorial129/images/"+ f"{doc_filename}-{page_no}.png"
            with open(filename,"wb")as f:
                page.image.pil_image.save(f,format="PNG")
            image_llm=llm.bind(images=[filename])
            response=image_llm.invoke("describe what you see in thge image , make simple , easy to understand and very clear in language , make it descriptive as much as you can , dont miss or skip any single detail")
            image_doc=Document(
            page_content=response,
            metadata={"source":"image-content","name":filename}
            )
            docs.append(image_doc)
    st=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=50)
    sd=st.split_documents(docs)
    vs=Chroma.from_documents(docs,embedding=embedding,persist_directory="./index")
    


    
