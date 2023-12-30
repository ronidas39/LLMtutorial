import requests
import json
import io
url="https://totaltechnology.atlassian.net/rest/api/3/search"
headers={
  "Accept": "application/json",
    "Content-Type": "application/json"
}

query = {
   'jql': 'project = TTS'
}

response=requests.get(url,headers=headers,params=query,auth=("ronidas071@gmail.com","wg5OpdxcSbGvlykzUhm5C939"))
data=response.json()
issues=data["issues"]
for issue in issues:
    issue_key=issue["key"]
    url1="https://totaltechnology.atlassian.net/rest/api/3/issue/"+issue_key
    response=requests.get(url1,headers=headers,auth=("ronidas071@gmail.com","wg5OpdxcSbGvlykzUhm5C939"))
    data=response.json()
    print(f'issue id is {issue_key} and status is {data["fields"]["status"]["name"]}')



