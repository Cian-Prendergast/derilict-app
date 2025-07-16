import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env
load_dotenv()

# Grab credentials and deployment details from environment
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Check required values
if not all([endpoint, api_key, deployment]):
    raise Exception("❌ Missing one or more required environment variables.")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

# Send a chat completion request
print("🤖 Sending test chat completion...")
response = client.chat.completions.create(
    model=deployment,  # deployment name, NOT model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you tell me a fun fact about architecture?"}
    ]
)

# Print the response
print("✅ Response:")
print(response.choices[0].message.content)
