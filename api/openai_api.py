import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL = "gpt-6-luna"


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0,
)


def get_response(prompt):
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": "none"},
        input=prompt,
    )

    return response.output_text