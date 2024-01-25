from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
import io,os,glob,base64,uuid
from PIL import Image
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.schema.messages import HumanMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
llm=ChatOpenAI(model="gpt-4-vision-preview",max_tokens=1024)
rpe=partition_pdf(filename="indian_Cushine.pdf",extract_images_in_pdf=True,infer_table_structure=True,chunking_strategy="title",
                  max_characters=4000,new_after_n_chhars=3800,combine_text_under_n_chars=2000,
                  extract_image_block_output_dir=os.getcwd()+"/images")

def image2base64(ip):
    with Image.open(ip) as image:
        buffer=io.BytesIO()
        image.save(buffer,format=image.format)
        img_str=base64.b64encode(buffer.getvalue())
        return img_str.decode("utf-8")
    
cwd="/Users/roni/Documents/GitHub/LLMtutorial/tutorial34/images"
files=glob.glob(cwd+"/*.jpg")
imgs=[]
imgs_summary=[]

for file in files:
    image_str=image2base64(file)
    response=llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type":"text","text":"please give a summary of the image provided , be descriptive and smart"},
                    {"type":"image_url","image_url":
                     {
                        "url":f"data:image/jpg;base64,{image_str}"

                    },
                    },
                ]
            )
        ]
    )
    imgs.append(image_str)
    imgs_summary.append(response.content)

te=[]
tes=[]
tbe=[]
tbes=[]
summary_prompt="""
summarize the following
{element_type}:
{element}
"""
summary_chain=LLMChain(
    llm=ChatOpenAI(model="gpt-4",max_tokens=1024),
    prompt=PromptTemplate.from_template(summary_prompt)
)

for e in rpe:
    if "CompositeElement" in repr(e):
        te.append(e.text)
        summary=summary_chain.run({"element_type":"text","element":e})
        tes.append(summary)
    elif "Table" in repr(e):
        tbe.append(e.text)
        summary=summary_chain.run({"element_type":"table","element":e})
        tbes.append(summary)
documents=[]
rc=[]
for e,s in zip(te,tes):
    i=str(uuid.uuid4())
    doc=Document(
        page_content=s,
        metadata={
            "id":i,
            "type":"text",
            "original_content":e
        }
    )
    rc.append((i,e))
    documents.append(doc)
for e,s in zip(tbe,tbes):
    i=str(uuid.uuid4())
    doc=Document(
        page_content=s,
        metadata={
            "id":i,
            "type":"table",
            "original_content":e
        }
    )
    rc.append((i,e))
    documents.append(doc)

for e,s in zip(imgs,imgs_summary):
    i=str(uuid.uuid4())
    doc=Document(
        page_content=s,
        metadata={
            "id":i,
            "type":"image",
            "original_content":e
        }
    )
    rc.append((i,e))
    documents.append(doc)

vs=FAISS.from_documents(documents=documents,embedding=OpenAIEmbeddings())
vs.save_local("cushine_index")