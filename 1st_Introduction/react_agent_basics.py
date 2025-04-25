from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain_community.tools import TavilySearchResults, tool
import os
import datetime

load_dotenv()

search_tool = TavilySearchResults(
    search_depth="basic", api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    return datetime.datetime.now().strftime(format)


tools = [search_tool, get_system_time]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
)

agent.invoke(
    "When was the SpaceX's last launch and how many days ago was that from this instant."
)

# print(llm.invoke("What is the capital of France?"))
