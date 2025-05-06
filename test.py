import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "lampu taman sudah banyak yang rusak"}

response = requests.post(url, json=data)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")