import requests

API_KEY="cm9uaWRhczA0MTk4N0BnbWFpbC5jb20:vU6fHQuEgIZK6zzLY8QK2"

def download_video(id):
    url = "https://api.d-id.com/talks/"+id

    headers = {
        "accept": "application/json",
        "authorization": f"Basic {API_KEY}"
    }

    response = requests.get(url, headers=headers)
    print(response.json())
    url=response.json()["result_url"]
    
    return url