
import os

from dotenv import load_dotenv
from mistralai.client import Mistral


load_dotenv()

MODEL = "mistral-small-2603"


client = Mistral(
    api_key=os.getenv("MISTRAL_API_KEY"),
)


def get_response(prompt):
    response = client.chat.complete(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        reasoning_effort="none",
    )

    return response.choices[0].message.content
