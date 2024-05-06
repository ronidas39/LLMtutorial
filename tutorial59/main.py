from langchain.utilities import SerpAPIWrapper
import os,requests
os.environ["SERPAPI_API_KEY"]="d002ee79b7052e74d6eb732592b959f84c758dc247d8437ea69b413fbf39513c"

params={
    "engine":"google_images"
}
search=SerpAPIWrapper(params=params)
results=search.run("mr olympia 2023 winner classic division")

data=results.replace("[","")
data=data.replace("]","")
data=data.replace("'","")
urls=data.split(",")

for url in urls:
    response=requests.get(url)
    if response.status_code==200:
        filename=url.split("/")[-1]
        with open(filename,"wb") as file:
            file.write(response.content)
            print(f"image download successfully: {filename}")
    else:
        print("failed to download")

