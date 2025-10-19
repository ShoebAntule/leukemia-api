import requests

# Test ping
response = requests.get("http://localhost:8000/ping")
print("Ping:", response.json())

# Test predict with cnn
files = {'file': open('Training/Blood cell Cancer [ALL]/Benign/Sap_013 (1).jpg', 'rb')}
response = requests.post("http://localhost:8000/predict?model=cnn", files=files)
print("CNN Predict:", response.json())

# Test predict with gru
files = {'file': open('Training/Blood cell Cancer [ALL]/Benign/Sap_013 (1).jpg', 'rb')}
response = requests.post("http://localhost:8000/predict?model=gru", files=files)
print("GRU Predict:", response.status_code, response.json() if response.status_code == 200 else response.text)
