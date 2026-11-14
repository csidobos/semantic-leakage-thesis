import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = client.models.retrieve("gpt-6-luna")
print(model.model_dump_json(indent=2))