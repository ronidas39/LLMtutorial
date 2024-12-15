import requests,json,io 
import string
alphabets=list(string.ascii_lowercase)
with io.open("urls.csv","w",encoding="utf-8")as f1:
    for alphabet in alphabets:
        url=f"https://www.1mg.com/pharmacy_api_gateway/v4/ayurvedas/by_alphabet?alphabet={alphabet}&page=1&per_page=50"
        response=requests.get(url)
        items=response.json()["data"]["schema"]["itemListElement"]
        for item in items:
            f1.write(item["name"]+","+item["url"]+"\n")
    f1.close()
