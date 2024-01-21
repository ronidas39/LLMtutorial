from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
import io,glob,os
from langchain.schema.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image
import base64
load_dotenv()
llm=ChatOpenAI(model="gpt-4-vision-preview",max_tokens=1024)
rpe=partition_pdf(filename="sample.pdf",extract_images_in_pdf=True,infer_table_structure=True,chunking_strategy="title",
                  max_characters=4000,new_after_n_chars=3800,combine_text_under_n_chars=2000)

def image2base64(image_path):
    with Image.open(image_path) as image:
        buffer=io.BytesIO()
        image.save(buffer,format=image.format)
        img_str=base64.b64encode(buffer.getvalue())
        return img_str.decode("utf-8")
    
cwd="/Users/roni/Documents/GitHub/LLMtutorial/tutorial33/figures"
files=glob.glob(cwd+"/*.jpg")
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
    print(file.split("/")[-1])
    print(response.content)
    print("========================================")
