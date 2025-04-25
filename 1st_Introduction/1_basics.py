from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor, tool
import datetime

load_dotenv()


@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

query = "what is the current time?"

prompt_template = hub.pull("hwchase17/react")

tools = [get_current_time]
agent = create_react_agent(
    llm,
    tools,
    prompt_template,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": query})
# print(result)
