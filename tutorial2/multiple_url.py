from langchain.document_loaders import youtube
import io
urls=["https://www.youtube.com/watch?v=iEDS_KjSF58","https://www.youtube.com/watch?v=tny_o4RmhvM","https://www.youtube.com/watch?v=eIBPZfls2sA"]
for url in urls:
    loader=youtube.YoutubeLoader.from_youtube_url(url)
    docs=loader.load()
    name=url.split("=")[1]
    with io.open(name+".txt","w",encoding="utf-8")as f1:
        for doc in docs:
            f1.write(doc.page_content)
        f1.close()