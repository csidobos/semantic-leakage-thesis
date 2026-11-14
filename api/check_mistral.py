import os
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

model = client.models.retrieve(model_id="mistral-small-2603")

print(model.model_dump_json(indent=2))
