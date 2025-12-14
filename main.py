from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any,Union,TypedDict
from langchain_core.messages import HumanMessage
from config import llm
from langchain_core.output_parsers import StrOutputParser

from agent_tool_calling import router_prompt, safety_prompt, draftsman_prompt, clinical_prompt
import json
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
import logging
import os,sys
from fastapi import FastAPI,HTTPException
from uvicorn import run
from pydantic import BaseModel
from postgres_connector import PostgresCheckpointer
from datetime import datetime

# class State(BaseModel):
#     user_input: str = ""
#     router_output: Optional[Dict[str, Any]] = None
#     safety_output: Optional[Dict[str, Any]] = None
#     draft_output: Optional[Dict[str, Any]] = None
#     critic_output: Optional[Dict[str, Any]] = None
#     final_output: Optional[str] = None

api=FastAPI()

class State(TypedDict):
    user_input:str
    task:str
    next_agent:str
    safety_result:bool
    result:dict
    final_result:str


def router_node(state: State) -> State:
    """Supervisor / Router that decides next agent."""
    # response = Router.invoke({"messages": [HumanMessage(content=state.user_input)]})
    print("user_query:",state["user_input"])
    chain=router_prompt | llm | StrOutputParser()
    print("pass")
    response:str= chain.invoke({"query":state["user_input"]})
    response=response.replace("```json","").replace("```","").strip()
    print("router response:",response)
    response=json.loads(response)
    state["task"] = response["payload"]
    state["next_agent"]=response["next_agent"]
    return state


def safety_node(state: State) -> State:
    """Runs SafetyGuardian check"""
    # payload = state.router_output.get("payload", "")
    chain=safety_prompt | llm | StrOutputParser()
    response = chain.invoke({"input":state["task"]})
    response=response.replace("```json","").replace("```","").strip()
    print("safety response:",response)
    response=json.loads(response)
    state["safety_result"] = response["safe"]
    if not response["safe"]:
        state["final_result"]=f"Crisis detected. Provide crisis resources and stop.{response["response_text"]}"
    else:
        state["next_agent"]="Draftsman"
    return state


def draftsman_node(state: State) -> State:
    """Creates structured CBT draft"""
    print("*****Entering the draft men******")
    # payload = state.router_output.get("payload", "")
    chain=draftsman_prompt | llm | StrOutputParser()
    response = chain.invoke({"input":state["task"]})
    response=response.replace("```json","").replace("```","").strip()
    print("draft response:",response)
    response=json.loads(response)
    state["result"]=response
    state["next_agent"]="ClinicalCritic"
    return state


def critic_node(state: State) -> State:
    """Reviews draft clinically"""
    # draft = state.draft_output.get("draft_text", "")
    chain=clinical_prompt | llm | StrOutputParser()
    response = chain.invoke({"input":state["result"]})
    response=response.replace("```json","").replace("```","").strip()
    print("clinical response:",response)
    response=json.loads(response)
    state["result"]=response
    return state


def finalize_node(state: State) -> State:
    """Supervisor generates final assembled CBT exercise."""
    # draft = state.draft_output.get("draft_text", "")
    # issues = state.critic_output.get("issues", [])
    # edits = state.critic_output.get("suggested_edits", "")
    final_result=state["result"]

    final_text = f"""
Here is your CBT Exercise:

{final_result}


"""
    state["final_result"] = final_text.strip()
    return state


def route_logic(state: State):
    """Router decides next agent based on router_output JSON."""

    nxt = state.get("next_agent")

    if nxt == "SafetyGuardian":
        return "safety"
    elif nxt == "Draftsman":
        return "draft"
    elif nxt == "ClinicalCritic":
        return "critic"
    else:
        return "finalize"


# checkpointer = MemorySaver()
checkpointer = PostgresCheckpointer(
        "postgresql://postgres:jeeva123@localhost:5432/mcp-server-cbt"
    )

graph = StateGraph(State)

# Nodes
graph.add_node("router", router_node)
graph.add_node("safety", safety_node)
graph.add_node("draft", draftsman_node)
graph.add_node("critic", critic_node)
graph.add_node("finalize", finalize_node)

# Start → Router
graph.set_entry_point("router")

# Supervisor decides next path
graph.add_conditional_edges(
    "router",
    route_logic,
    {
        "safety": "safety",
        "draft": "draft",
        "critic": "critic",
        "finalize": "finalize"
    }
)

# graph.add_edge("safety", "router")
# graph.add_edge("draft", "router")
# graph.add_edge("critic", "router")
# graph.add_edge("finalize", END)

graph.add_conditional_edges(
    "safety",
    lambda state: state.get("next_agent", None),
    {"Draftsman": "draft"},
    # if no next_agent → safety blocked => end
)

graph.add_edge("draft", "critic")
graph.add_edge("critic","finalize")
graph.add_edge("finalize",END)



app = graph.compile(checkpointer=checkpointer)

config={
    "configurable":
    {
        "thread_id":"user-123"
    }
}

# query:str="give me a better a sleeping schedule"

# result=app.invoke({"user_input": query},
#     config=config)

# print("result:",result["final_result"])

class User(BaseModel):
    user_input:str


def format_cbt_result(result: Dict) -> str:
    """
    Format the CBT result dictionary into a readable string
    """
    if isinstance(result, str):
        return result
    
    # Format based on your specific result structure
    formatted = ""
    
    if "title" in result:
        formatted += f"# {result['title']}\n\n"
    
    if "overview" in result:
        formatted += f"## Overview\n{result['overview']}\n\n"
    
    if "steps" in result:
        formatted += "## Steps\n"
        for i, step in enumerate(result["steps"], 1):
            formatted += f"{i}. {step}\n"
        formatted += "\n"
    
    if "coping_strategies" in result:
        formatted += f"## Coping Strategies\n{result['coping_strategies']}\n\n"
    
    if "notes" in result:
        formatted += f"## Notes\n{result['notes']}\n"
    
    # Fallback: just convert to string
    if not formatted:
        formatted = str(result)
    
    return formatted



@api.post("/mcp-chat")
async def chat_with_mcp(question:User):
    try:
        user_input=question.user_input
        result=app.invoke({"user_input":user_input},config=config)
        return {"response":f"{result["final_result"]}"}
    except Exception as ex:
        raise HTTPException(status_code=400,detail=ex)

@api.get("/workflow-state/{thread_id}")
async def get_workflow_state(thread_id: str):
    """
    GET endpoint to fetch current workflow state from PostgreSQL checkpoint
    
    Returns:
        - steps: List of agent steps executed
        - current_state: Current workflow state (idle, running, awaiting_approval, etc.)
        - awaiting_approval: Boolean indicating if workflow is halted
        - checkpoint_state: Current checkpoint data if awaiting approval
    """
    try:
        # Fetch checkpoint from PostgreSQL
        checkpoint_tuple = checkpointer.get_tuple({
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ""
            }
        })
        
        if not checkpoint_tuple:
            return {
                "steps": [],
                "current_state": "idle",
                "awaiting_approval": False,
                "checkpoint_state": None
            }
        
        checkpoint_data = checkpoint_tuple.checkpoint
        metadata = checkpoint_tuple.metadata
        print(f"checkpoint_data:{checkpoint_data}")
        # Extract state information from checkpoint
        state = checkpoint_data.get("channel_values", {})
        print(f"state:{state}")

        
        # Build agent steps from checkpoint history
        steps = []
        
        # Check which agents have executed based on state
        if state.get("task"):
            steps.append({
                "id": 1,
                "agent": "Router",
                "icon": "🎯",
                "action": "Analyzed user request",
                "thought": f"Identified task: {state.get('task', 'N/A')}",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            })
        
        if state.get("safety_result") is not None:
            steps.append({
                "id": 2,
                "agent": "SafetyGuardian",
                "icon": "🛡️",
                "action": "Safety check completed",
                "thought": "Content is safe" if state["safety_result"] else "Safety concerns detected",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            })
        
        if state.get("result"):
            steps.append({
                "id": 3,
                "agent": "Draftsman",
                "icon": "✍️",
                "action": "Created CBT exercise draft",
                "thought": "Generated structured CBT content",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            })
            
            steps.append({
                "id": 4,
                "agent": "ClinicalCritic",
                "icon": "🩺",
                "action": "Reviewed clinical accuracy",
                "thought": "Content follows CBT best practices",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            })
        
        # Determine current state and if awaiting approval
        awaiting_approval = False
        current_state = "running"
        
        # Check if workflow is at the approval stage
        if state.get("result") and not state.get("final_result"):
            awaiting_approval = True
            current_state = "awaiting_approval"
        elif state.get("final_result"):
            current_state = "completed"
            steps.append({
                "id": 5,
                "agent": "Finalize",
                "icon": "🎁",
                "action": "Finalized CBT exercise",
                "thought": "Assembled final deliverable",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            })
            steps.append({
                "id": 6,
                "agent": "PostgreSQL",
                "icon": "💾",
                "action": "Saved to checkpoint",
                "thought": "Persisted to database",
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            })
        
        # Prepare checkpoint state for approval
        checkpoint_state = None
        if awaiting_approval:
            result = state.get("result", {})
            # Convert result dict to readable string if needed
            if isinstance(result, dict):
                result_str = format_cbt_result(result)
            else:
                result_str = str(result)
            
            checkpoint_state = {
                "checkpoint_id": checkpoint_tuple.config["configurable"]["checkpoint_id"],
                "result": result_str,
                "draft": result_str,
                "timestamp": datetime.now().isoformat(),
                "full_state": state
            }
            print(f"checkpointer:{checkpoint_state}")
        
        return {
            "steps": steps,
            "current_state": current_state,
            "awaiting_approval": awaiting_approval,
            "checkpoint_state": checkpoint_state
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching workflow state: {str(e)}")


# name = "cbt-agent-mcp"
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(name)
# port = int(os.environ.get('PORT', 8000))


# mcp = FastMCP(name, port=port)

# @mcp.tool()
# async def run_cbt_pipeline(user_input:str)->TextContent:
#     result=app.invoke({"user_input":user_input},config=config)
#     return TextContent(result["final_result"])


# if __name__=="__main__":
#     logger.info(f"Starting MCP Server on port {port}...")
#     try:
#         mcp.run(transport="streamable-http")
#     except Exception as e:
#         logger.error(f"Server error: {str(e)}")
#         sys.exit(1)
#     finally:
#         logger.info("Server terminated")

if __name__=="__main__":
    run("main:api",
        host="127.0.0.1",
        port=8001,
        reload=False)