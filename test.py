import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is missing. Please check your environment or .env file.")

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "llama-3.3-70b-versatile",

    "messages": [
        {
            "role": "user",
            "content": "Explain the importance of fast language models"
        }
    ]
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=60
)

print(f"Status Code: {response.status_code}")

if response.ok:

    print(response.json())

else:

    print(response.text)