import requests

def getUrl(id,auth_token):
    while(1):
        url = "https://cloud.leonardo.ai/api/rest/v1/generations/"+id

        headers = {
            "accept": "application/json",
            "authorization": "Bearer "+auth_token
        }

        response = requests.get(url, headers=headers)

        urls=response.json()['generations_by_pk']['generated_images']
        if len(urls)>0:
            return urls[0]["url"]
