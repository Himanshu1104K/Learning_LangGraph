import json
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_community.tools import TavilySearchResults

tavily_search = TavilySearchResults(max_results=5)


def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message = state[-1]

    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []

    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_query = tool_call["args"].get("search_queries", [])

            query_results = {}

            for query in search_query:
                result = tavily_search.invoke(query)
                query_results[query] = result

            tool_messages.append(
                ToolMessage(content=json.dumps(query_results), tool_call_id=call_id)
            )

    return tool_messages
