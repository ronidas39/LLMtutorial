from lumaai import LumaAI
import time
client=LumaAI()

def generateVideo(input):
    response=client.generations.create(prompt=input)
    id=response.id
    video_metadata=client.generations.get(id=id)
    print(id,video_metadata)