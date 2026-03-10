from typing import Annotated, TypedDict, Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

from prompts import (
    SUPERVISOR_PROMPT,
    DATA_INSPECTOR_PROMPT,
    ANOMALY_DETECTOR_PROMPT,
    DEGRADATION_ANALYST_PROMPT)

from tools import (
    get_engines_set_summary,
    get_engine_stats,
    get_critical_engines,
    detect_anomalies,
    compare_engines,
    get_sensor_trend)

# ---------------------------------------------------------------------------
# 1. Shared State running through the whole graph
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str       #question typed by the user
    route: str          #supervisor decision - which agent answers
    answer: str         #answer for the user
    messages: Annotated[list[BaseMessage], add_messages] #chat history

# ---------------------------------------------------------------------------
# 2. Creating agent functions
# ---------------------------------------------------------------------------

def _build_agent_executor(llm, prompt: str, tools: list):
    agent = create_react_agent(model=llm, prompt=prompt,tools=tools)
    return agent

def _extract_message(result: dict) -> str:
    messages = result.get("messages", [])
    #Search for the last message from AI agent (AIMessage type)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    #No AIMessage, try any content
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            return msg.content
    #No message content
    return result.get("answer") or "No answer found"

# ---------------------------------------------------------------------------
# 3. Creating graph nodes for each agent
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState, llm) -> AgentState:
    """
    Read the question -> decide which agent anwers the specific question
    No tools, only routing
    """
    #Create chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        MessagesPlaceholder(variable_name="history"), #remeber chat history 
        ("human", "{question}") #question placeholder
    ])
    
    #Call model history aware
    chain = prompt | llm
    response = chain.invoke({"history": state.get("messages", []), "question": state["question"]})

    #Clean content
    raw_content = response.content.strip().lower()

    #Select agent
    valid_agents = ["data_inspector", "anomaly_detector", "degradation_analyst"]
    route = next((agent for agent in valid_agents if agent in raw_content), "data_inspector")  #return first macth (next)
    print(f"[Supervisor] routing to: [{route}]")

    return {"route": route, "messages": [HumanMessage(content=state["question"])]}

def data_inspector_node(state: AgentState, llm) -> AgentState:
    """
    Agent spcializing in general statistics concerning genral engine information and data stucture
    Tools: get_enignes_set_summary, get_engine_stats
    """
    #Create data inspector agent
    executor = _build_agent_executor(llm, DATA_INSPECTOR_PROMPT, tools=[get_engines_set_summary, get_engine_stats])
    #Send task to agent - history aware
    current_messages = state["messages"] + [HumanMessage(content=state["question"])]
    result = executor.invoke({"messages": current_messages})
    new_messages = result.get("messages", [])

    return {"answer": _extract_message(result), "messages": new_messages}

def anomaly_detector_node(state: AgentState, llm) -> AgentState:
    """
    Agent spcializing in detecting anomalies and identifying faulty engines
    Tools: get_critical_engines, detect_anomalies
    """
    #Create data inspector agent
    executor = _build_agent_executor(llm, ANOMALY_DETECTOR_PROMPT, tools=[get_critical_engines, detect_anomalies])
    #Send task to agent - history aware
    current_messages = state["messages"] + [HumanMessage(content=state["question"])]
    result = executor.invoke({"messages": current_messages})
    new_messages = result.get("messages", [])

    return {"answer": _extract_message(result), "messages": new_messages}

def degradation_analyst_node(state: AgentState, llm) -> AgentState:
    """
    Agent spcializing in degradation analysis, engines comparison and trends descriptions across crucial sensors
    Tools: compare_engines, get_sensor_trend
    """
    #Create data inspector agent
    executor = _build_agent_executor(llm, DEGRADATION_ANALYST_PROMPT, tools=[compare_engines, get_sensor_trend])
    #Send task to agent - history aware
    current_messages = state["messages"] + [HumanMessage(content=state["question"])]
    result = executor.invoke({"messages": current_messages})
    new_messages = result.get("messages", [])

    return {"answer": _extract_message(result), "messages": new_messages}

# ---------------------------------------------------------------------------
# 4. Routing – supervisor decision - possibel graph routes
# ---------------------------------------------------------------------------

def route_question(state: AgentState) -> Literal["data_inspector", "anomaly_detector", "degradation_analyst"]:
    return state["route"]


# ---------------------------------------------------------------------------
# 5. Graph structure
# ---------------------------------------------------------------------------

def build_graph(llm):

    graph = StateGraph(AgentState)

    #Lambda state add llm to each node
    graph.add_node("supervisor", lambda s: supervisor_node(s,llm))
    graph.add_node("data_inspector", lambda s: data_inspector_node(s, llm))
    graph.add_node("anomaly_detector", lambda s: anomaly_detector_node(s, llm))
    graph.add_node("degradation_analyst", lambda s: degradation_analyst_node(s, llm))

    #Start with supervisor
    graph.set_entry_point("supervisor")

    #After supervisorze – conditional edge based on state["route"]
    graph.add_conditional_edges(
        "supervisor",
        route_question, {"data_inspector": "data_inspector", "anomaly_detector": "anomaly_detector", "degradation_analyst": "degradation_analyst"})
    
    #Each agent is the graph endpoint
    graph.add_edge("data_inspector", END)
    graph.add_edge("anomaly_detector", END)
    graph.add_edge("degradation_analyst", END)

    return graph.compile()

# ---------------------------------------------------------------------------
# 6. MAIN FUNCTION including chat history – connected to app.py
# ---------------------------------------------------------------------------

def run_agent(question: str, llm, chat_history: list) -> tuple[str, list]:

    graph = build_graph(llm)
    initial_state: AgentState = {"question": question, "route": "", "answer": "", "messages": chat_history or []}
    result = graph.invoke(initial_state)
    return result["answer"], result["messages"]



