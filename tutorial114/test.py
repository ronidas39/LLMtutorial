from lumaai import LumaAI
import requests
import time
client=LumaAI()
video_metadata=client.generations.get(id="3b3ef40c-5cfb-4d0e-a4d1-63b4b132c5bd")
print(video_metadata.state)
# link=video_metadata.assets.video
# response=requests.get(link,stream=True)
# file_name="video.mp4"
# with open(file_name,"wb")as f1:
#     f1.write(response.content)
    
