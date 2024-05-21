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
   response=requests.post(url,headers=headers,data=payload,auth=(""))
   data=response.json()
   return data


