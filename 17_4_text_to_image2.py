# headers = {"Authorization": "Bearer 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"}
import requests

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
image_bytes = query({
	"inputs": "Astronaut riding a horse",
})
# You can access the image with PIL.Image for example
# print(image_bytes)
import io
from PIL import Image

image = Image.open(io.BytesIO(image_bytes))
image.show()