import requests

url = "https://cloud.leonardo.ai/api/rest/v1/generations"


def genImage(ids,auth_token):
    payload = {
        "alchemy": True,
        "height": 768,
        "modelId": "b24e16ff-06e3-43eb-8d33-4416c2d75876",
        "num_images": 1,
        "presetStyle": "DYNAMIC",
        "prompt": ids,
        "width": 1024
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer "+auth_token
    }

    response = requests.post(url, json=payload, headers=headers)

    response=response.json()["sdGenerationJob"]["generationId"]
    return response