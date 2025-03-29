#from langchain.document_loaders import youtube
from langchain_community.document_loaders import YoutubeLoader
import io

loader=YoutubeLoader.from_youtube_url("https://youtu.be/79kidqE_nu8")
docs=loader.load()
print(docs)
with io.open("transcript.txt","w",encoding="utf-8")as f1:
    for doc in docs:
        f1.write(doc.page_content)
    f1.close()