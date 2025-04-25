from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


class Country(BaseModel):
    """Information about a country"""

    name: str = Field(description="The name of the country")
    languages: str = Field(description="The languages spoken in the country")
    capital: str = Field(description="The capital city of the country")


structured_llm = llm.with_structured_output(Country)

response = structured_llm.invoke("Tell me about France?")
print(response)
