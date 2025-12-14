from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from config import llm
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate




load_dotenv()

llm = llm
checkpointer = MemorySaver()

router_prompt =ChatPromptTemplate.from_template( """
You are the ROUTER (Supervisor).

Your role:
- Read the user query and decide or analyze the workflow according to the user query .
- Route work to the right CBT agents.
- Decide the workflow: Safety → Draft → Clinical Review → Finalize.
- Return a JSON-like structured message describing which agent should act next
  and what they should do.
- Generate your response in only in valid Json and dont generate any extra paragraph or words part from json.
- You should only give next_agent as SafetyGuardian

Do NOT generate CBT content yourself.
user query: {query}
Give me a valid json Format as response:
```json
{{
 "next_agent": "SafetyGuardian" ,
 "payload": "<content you want that agent to process>"
}}
```
""")

Router = create_react_agent(
    model=llm,
    tools=[],                # no tools
    checkpointer=checkpointer,
    prompt=router_prompt
)

safety_prompt =ChatPromptTemplate.from_template( """
You are the SafetyGuardian Agent and you should analyze, make decision according to the detail that is provided.

Your task:
- Examine the user's request for suicide risk, self-harm, violence, medical emergency.
- If unsafe: return a supportive crisis-safe response + mark "unsafe".
- If safe: mark "safe".
- Generate your response in only in valid Json and dont generate any extra paragraph or words part from json.

Here is the detail: {input}

Give me Valid Json Format as response:
```json
{{
 "safe": true | false,
 "response_text": "<If unsafe, provide safety response. If safe, leave empty>"
}}```
""")

SafetyGuardian = create_react_agent(
    model=llm,
    tools=[],
    checkpointer=checkpointer,
    prompt=safety_prompt
)

draftsman_prompt =ChatPromptTemplate.from_template( """
You are the Draftsman Agent and you should analyze, make decision according to the detail that is provided.

Your task:
- Create a structured first-draft CBT exercise based on the user's intent.
- Keep content safe, structured, and clinically aligned.
- Generate your response in only in valid Json and dont generate any extra paragraph or words part from json.
- json should be correct and should be strickly following json formate where it should be containing any special character such as symbols.

Here is the detail: {input}

Give me Valid Json Format as response:
```json
{{
 "draft_text": "<the CBT exercise draft>"
}}```
""")

Draftsman = create_react_agent(
    model=llm,
    tools=[],
    checkpointer=checkpointer,
    prompt=draftsman_prompt
)

clinical_prompt = ChatPromptTemplate.from_template("""
You are the ClinicalCritic Agent and you should analyze, make decision according to the detail that is provided.

Your task:
- Review the Draftsman's output for tone, empathy, clarity, safety.
- Suggest corrections to improve clinical quality.
- DO NOT rewrite the full content unless necessary.
- Generate your response in only in valid Json and dont generate any extra paragraph or words part from json.

Here is the detail:{input}

Give me Valid Json Format as response:
```json
{{
 "score": <0-100>,
 "issues": [...],
 "suggested_edits": "<text with recommended improvements>"
}}
```
""")

ClinicalCritic = create_react_agent(
    model=llm,
    tools=[],
    checkpointer=checkpointer,
    prompt=clinical_prompt
)



