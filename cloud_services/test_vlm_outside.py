from PIL import Image
import requests
import base64
from io import BytesIO

def convert_pil_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

class LLaVA():
    def __init__(self, server_url="http://werewolf-key-subtly.ngrok-free.app/llava_chat", temperature=0.0):
        self.server_url = server_url
        self.temperature = temperature

    def chat(self, prompt, image, meta_prompt=""):
        base64_image = convert_pil_image_to_base64(image)
        payload = {
            "prompt": meta_prompt + '\n' + prompt,
            "image": base64_image,
            "max_new_tokens": 2048,
            "temperature": self.temperature
        }
        response = requests.post(
            self.server_url, 
            json=payload,
        ).json()
        return response["text"]


client = LLaVA()
"""Test the LLaVA chat function with an image and a prompt."""
url = "https://www.californiastrawberries.com/wp-content/uploads/2021/05/Rainbow-Fruit-Salad-1024-500x375.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "How many type of fruits are in the image?"

response = client.chat(prompt, image)
print("\nLLaVA responded:\n", response, '\n', '_'*50)