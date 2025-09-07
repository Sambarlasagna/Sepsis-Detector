import requests
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY") 
url = f"https://generativelanguage.googleapis.com/v1/models?key={API_KEY}"

response = requests.get(url)

if response.status_code == 200:
    models = response.json()
    print(models)
else:
    print(f"Error: {response.status_code} - {response.text}")
