# The 'requests' and 'json' libraries are imported. 
# 'requests' is used to send HTTP requests, while 'json' is used for parsing the JSON data that we receive from the API.
import requests
import json,os

# An API key is defined here. You'd normally get this from the service you're accessing. It's a form of authentication.
XI_API_KEY = "<xi-api-key>"

# This is the URL for the API endpoint we'll be making a GET request to.
url = "https://api.elevenlabs.io/v1/voices"

# Here, headers for the HTTP request are being set up. 
# Headers provide metadata about the request. In this case, we're specifying the content type and including our API key for authentication.
headers = {
  "Accept": "application/json",
  "xi-api-key":  os.getenv("ELEVEN_API_KEY"),
  "Content-Type": "application/json"
}

# A GET request is sent to the API endpoint. The URL and the headers are passed into the request.
response = requests.get(url, headers=headers)

# The JSON response from the API is parsed using the built-in .json() method from the 'requests' library. 
# This transforms the JSON data into a Python dictionary for further processing.
data = response.json()

# A loop is created to iterate over each 'voice' in the 'voices' list from the parsed data. 
# The 'voices' list consists of dictionaries, each representing a unique voice provided by the API.
for voice in data['voices']:
  # For each 'voice', the 'name' and 'voice_id' are printed out. 
  # These keys in the voice dictionary contain values that provide information about the specific voice.
  print(f"{voice['name']}; {voice['voice_id']}")
