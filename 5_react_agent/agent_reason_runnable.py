from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_react_agent, tool
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
search_tool = TavilySearchResults(search_depth="basic")


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    return datetime.datetime.now().strftime(format)


tools = [search_tool, get_system_time]

react_prompt = hub.pull("hwchase17/react")
react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)
