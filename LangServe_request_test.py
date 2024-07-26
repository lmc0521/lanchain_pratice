import requests


headers={'Authorization': 'Bearer valid_api_key'}
response = requests.post(
    "http://localhost:9000/llama2/invoke",
    json={'input': {'input': 'what is my name?'}},
    headers=headers
)
print(response.json())