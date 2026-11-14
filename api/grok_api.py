
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "grok-4.20-0309-non-reasoning"

client = OpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1",
    timeout=60.0,
    max_retries=0,
)


def get_response(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content
