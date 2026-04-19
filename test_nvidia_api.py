import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
print(f"✅ API Key set: {bool(NVIDIA_API_KEY)}")
print(f"✅ Key length: {len(NVIDIA_API_KEY)}")

if not NVIDIA_API_KEY:
    print("❌ NVIDIA_API_KEY not found. Set it in .env or environment.")
    exit(1)

try:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )
    
    response = client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": "Say 'Hello'"}],
        max_tokens=10,
    )
    
    print("✅ NVIDIA API is working!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ Error: {e}")