import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

# Load Azure credentials
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-image-1")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

# Construct full URL
if not endpoint.endswith("/"):
    endpoint += "/"
url = f"{endpoint}openai/deployments/{deployment}/images/edits?api-version={api_version}"

# Prompt for restoration
prompt = (
    "Restore this derelict building: clean the brickwork, repair windows, "
    "add fresh paint, modern lighting, and surrounding greenery. Keep the structure intact."
)

# Read the image file
image_path = "derelict-site.png"
if not os.path.exists(image_path):
    raise FileNotFoundError("❌ 'derelict-site.png' not found in current directory.")

with open(image_path, "rb") as image_file:
    files = {
        "image[]": (image_path, image_file, "image/png")
    }

    data = {
        "prompt": prompt,
        "model": deployment,
        "size": "1024x1024",
        "quality": "high",
        "n": 1
    }

    headers = {
        "api-key": api_key
    }

    print("🎨 Sending image to Azure OpenAI for editing...")

    response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        b64_img = response.json()["data"][0]["b64_json"]
        output_path = "derelict-site-restored.png"
        with open(output_path, "wb") as out_file:
            out_file.write(base64.b64decode(b64_img))
        print(f"✅ Restored image saved as: {output_path}")
    else:
        print(f"❌ Error {response.status_code}:")
        print(response.json())
