from PIL import Image
from apis.language_model import LLaVA
import requests
client = LLaVA()

"""Test the LLaVA chat function with an image and a prompt."""
# prompt = "where is the fruits?"
# image = Image.open("images/3.jpg")  # Adjust path as necessary

url = "https://www.californiastrawberries.com/wp-content/uploads/2021/05/Rainbow-Fruit-Salad-1024-500x375.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "How many type of fruits are in the image?"
response = client.chat(prompt, image)


# assert isinstance(response, str)  # Ensure the response is a string

print("\nLLaVA responded:\n", response, '\n', '_'*50)

