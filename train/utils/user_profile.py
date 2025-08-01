import openai
from openai import OpenAI
import os

openai.api_key = "[Your OpenAI API Key Here]"

client = OpenAI(
    api_key = openai.api_key
)

def get_openai_response(prompt):
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature = 0.7,
        max_tokens = 1024
    )

    return response.choices[0].message.content

USER_PROFILE_TEMPLATE = """

"""