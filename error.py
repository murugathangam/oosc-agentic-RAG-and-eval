import requests
with open("doc.txt", "r", encoding="utf-8") as f:
    material = f.read()
response = requests.post("http://localhost:8000/chunk1", json={"material": material, "chunk_size": 200, "overlap": 50})
print(response.json())