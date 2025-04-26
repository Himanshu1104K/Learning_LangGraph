from dotenv import load_dotenv
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import END, StateGraph

from agent_reason_runnable import react_agent_runnable, tools
from react_state import AgentState


load_dotenv()


def reason_node(state: AgentState):
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


def act_node(state: AgentState):
    agent_action = state["agent_outcome"]

    tool_name = agent_action.tool
    tool_input = agent_action.tool_input

    tool_function = None
    for tool in tools:
        if tool.name == tool_name:
            tool_function = tool
            break

    if tool_function:
        if isinstance(tool_input, dict):
            output = tool_function.invoke(**tool_input)
        else:
            output = tool_function.invoke(tool_input)
    else:
        output = f"Tool {tool_name} not found"
    return {"intermediate_steps": [(agent_action, str(output))]}


def filter_memory_node(state: AgentState):
    """Limit memory by keeping only the last 8 messages in intermediate_steps."""
    if len(state["intermediate_steps"]) > 8:
        state["intermediate_steps"] = state["intermediate_steps"][-8:]
    return {"intermediate_steps": state["intermediate_steps"]}


def should_continue(state: AgentState):
    """Check if we should continue or end the agent loop."""
    # If the agent outcome is None, we're just starting
    if state["agent_outcome"] is None:
        return "reason"
    
    # If the agent output is AgentFinish, we're done
    from langchain_core.agents import AgentFinish
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    
    # Otherwise, we continue with the loop
    return "reason"


# Create the agent graph
def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("reason", reason_node)
    workflow.add_node("act", act_node)
    workflow.add_node("filter_memory", filter_memory_node)
    
    # Build graph edges
    workflow.add_edge("reason", "act")
    workflow.add_edge("act", "filter_memory")
    workflow.add_edge("filter_memory", should_continue)
    
    # Set entry point
    workflow.set_entry_point("reason")
    
    return workflow.compile()


# Create the runnable agent graph
agent_graph = build_agent_graph()
