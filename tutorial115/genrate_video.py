from lumaai import LumaAI
import time
client=LumaAI()

def generateVideo(input):
    response=client.generations.create(prompt=input)
    id=response.id
    while(1):
        metadata=client.generations.get(id=id)
        print(metadata)
        if metadata.state=="completed":
            return id
        else:
            time.sleep(15)


def extendVideo(input, id):
    response=client.generations.create(
        prompt=input,
        keyframes={
            "frame0":{
                "type":"generation",
                "id":id
            }
        }
    )
    id=response.id
    while(1):
        metadata=client.generations.get(id=id)
        print(metadata)
        if metadata.state=="completed":
            return id
        else:
            time.sleep(15)


