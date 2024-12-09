# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI


ds_client = OpenAI(api_key="sk-4c723af969454bb790f889226a818c23", base_url="https://api.deepseek.com")
ds_client = OpenAI(api_key="sk-4c723af969454bb790f889226a818c23", base_url="https://api.deepseek.com")

# response = ds_client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )
#
# print(response.choices[0].message.content)