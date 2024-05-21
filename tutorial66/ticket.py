import requests
import json
url="https://thetotaltechnology.atlassian.net/rest/api/2/issue"
headers={
    "Accept": "application/json",
    "Content-Type": "application/json"
}
def generate_ticket(data):
   payload=json.dumps(
    {
    "fields": {
       "project":
       {
          "id": "10000"
       },
       "summary": "created for test11",
       "description": "Created for test11",
       "issuetype": {
          "name": "Task"
       }
      }
   }
   )
   response=requests.post(url,headers=headers,data=payload,auth=("thetotaltechnology@gmail.com","ATATT3xFfGF0oyzRgUnRJE9kwn6mGmkP5o0-0Wh7gcnce4fH4MzZbrGr4v--d3sam9b1EBeHILuP4REqm9bJbuWAnXFDug3qGh95wzImPOno88NNm-B_Ejo-TtreCiw5npLQjAZJFZH944LOVT0jxSqEgEG8rQOHUj9ZHwPPFqQZ7x7CaHh7D0M=F954B576"))
   data=response.json()
   return data


