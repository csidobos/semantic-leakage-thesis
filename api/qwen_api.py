import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL = "qwen3.8-flash"


client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    timeout=30.0,
)


def get_response(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort="none",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content