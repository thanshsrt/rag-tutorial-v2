import requests
import json

url = "http://127.0.0.1:5000/query"
payload = {"question": "What is this project about?"}

response = requests.post(url, json=payload, stream=True)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith('data: '):
                print(decoded[6:], end='', flush=True)
else:
    print(f"Error: {response.status_code} - {response.text}")