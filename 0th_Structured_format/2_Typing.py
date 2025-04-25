from typing_extensions import Annotated, TypedDict
from typing import Optional
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, 1 to 10"]


structured_llm = llm.with_structured_output(Joke)

response = structured_llm.invoke("Tell me a joke about a cats")
print(response)
