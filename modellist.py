from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

models = client.models.list()
for m in models.data:
    print(m.id)