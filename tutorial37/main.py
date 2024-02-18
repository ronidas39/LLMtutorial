from openai import OpenAI
import requests
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="a tatto design using some motivation quotes",
  size="1024x1024",
  quality="hd",
  n=1,
)

image_url = response.data[0].url



# Use the requests library to fetch the image content
response = requests.get(image_url)

# Check if the request was successful
if response.status_code == 200:
    # Open a file in binary write mode
    with open("siamese_cat_image.png", "wb") as file:
        # Write the content of the response to the file
        file.write(response.content)
    print("Image successfully saved.")
else:
    print("Failed to fetch image.")
