import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1",
)

model = client.models.retrieve("grok-4.20-0309-non-reasoning")

print(model.model_dump_json(indent=2))