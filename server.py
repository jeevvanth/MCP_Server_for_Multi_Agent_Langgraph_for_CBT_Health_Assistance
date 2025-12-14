from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
import httpx,requests
import json

# Create an MCP server
mcp = FastMCP("mcp-multi-agent-cbt", json_response=True)

config={
    "configurable":
    {
        "thread_id":1
    }
}



@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def run_cbt_pipeline(user_input:str)->TextContent:
    """
Health Assistance and Cognitive Behavioral Therapy
    
This tool generates supportive and structured Cognitive Behavioral Therapy (CBT)
exercises based on a user’s request (for example, exposure hierarchies or coping
strategies).

It uses an internal multi-agent workflow to:
in this we're going to call the api endpoint and what happen inside api operation: 
- Understand the user’s intent
- Check for safety concerns
- Create a clear and empathetic CBT exercise
- Review the content for tone and clarity

This tool does not provide medical advice or diagnosis.
If a crisis or self-harm risk is detected, it returns supportive guidance
instead of a CBT exercise.

Input:
- user_input (str): A brief description of the CBT task

Output:
- A structured, easy-to-follow CBT exercise in plain language
"""

    payload={"user_input":user_input}
    result=requests.post(url="http://127.0.0.1:8001/mcp-chat",json=payload)
    res=result.json()
    response_text=res.get("response",str(res))
    return TextContent(type="text",text=response_text)

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


# Add a prompt
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """Generate a greeting prompt"""
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }

    return f"{styles.get(style, styles['friendly'])} for someone named {name}."




# Run with streamable HTTP transport
if __name__ == "__main__":
    mcp.run(transport="streamable-http")