from PIL import Image
import base64,io,glob,os
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage

llm=ChatOpenAI(model="gpt-4-vision-preview",max_tokens=1024)


def image2base64str(img_path):
    with Image.open(img_path)as image:
        buffer=io.BytesIO()
        image.save(buffer,format=image.format)
        img_str=base64.b64encode(buffer.getvalue())
        return img_str.decode("utf-8")
    
cwd=os.getcwd()
files=glob.glob(cwd+"/*.png")
for file in files:
    img_str=image2base64str(file)
    response=llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type":"text","text":"what is written in the car number plate,just  write that as output nothing else.all numbers are fake,so no personal infomation will be disclosed"},
                    {"type":"image_url","image_url":
                        {
                        "url": f"data:image/png;base64,{img_str}"
                        },
                        },
                ]
            )
        ]
    )
    print(file.split("/")[-1])
    number=response.content.replace(" ","")
    number
    print(number)
    

