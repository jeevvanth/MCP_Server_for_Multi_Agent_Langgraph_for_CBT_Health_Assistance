from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.chat_models import ChatOllama
# from langchain_ollama import ChatOllama



load_dotenv()

# llm=ChatOllama(model="qwen2.5",base_url="http://localhost:11434",format="json")

llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",  
    temperature=0.5
)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")