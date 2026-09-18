import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    print("❌ MISTRAL_API_KEY not found")
    exit()

print("✅ MISTRAL_API_KEY found")

llm = ChatMistralAI(
    model="codestral-latest",
    temperature=0,
    api_key=api_key
)

try:
    response = llm.invoke(
        "Say hello and tell me that the Mistral API is working."
    )

    print("\n✅ Mistral API is working!")
    print("\nResponse:")
    print(response.content)

except Exception as e:
    print("\n❌ Mistral API request failed!")
    print("\nError:")
    print(e)