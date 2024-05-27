import requests ,json

url="https://thetotaltechnology.atlassian.net/rest/api/2/issue"
headers={
    "Accept":"application/json",
    "Content-Type":"application/json"
}
def gen_ticket(data):
    payload=json.dumps(
        {
            "fields":{
                "project":{
                    "id":"10000",

            },
            "summary":data["Title"],
            "description":data["Summary"],
            "issuetype":{
             "name":"Task"
            }

        }
        }
    )
    response=requests.post(url,headers=headers,data=payload,auth=("thetotaltechnology@gmail.com","xxxC"))
    data=response.json()
    return data