import requests

url = "http://127.0.0.1:8000/process_pdf/"
files = {'file': open(r'D:\LabAI\OCR\MinerUN\MinerU\pdfs\g.pdf', 'rb')}
try:
    response = requests.post(url, files=files)
    response.raise_for_status()
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    print(f"Response text: {response.text}")
finally:
    files['file'].close()