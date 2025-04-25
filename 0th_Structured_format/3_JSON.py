from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

json_schema = {
    "title": "joke",
    "description": "A joke to tell user",   
    "type": "object",
    "properties": {
        "setup": {"type": "string", "description": "The setup of the joke"},
        "punchline": {"type": "string", "description": "The punchline of the joke"},
        "rating": {"type": "number", "description": "How funny the joke is, 1 to 10"},
    },
    "required": ["setup", "punchline"],
}

structured_llm = llm.with_structured_output(json_schema)

response = structured_llm.invoke("Tell me a joke about a cat")
print(response)
