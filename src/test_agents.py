import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from agents import run_agent

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

hf_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="conversational",
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token)
llm = ChatHuggingFace(llm=hf_llm)

print("\n=== TEST 1: ===")
question1 = "Give me the summary of the entire dataset"
print(question1)
answer, history = run_agent(question1, llm, [])
print(answer)

print("\n=== TEST 2: ===")
question2 = "Are there any anomalies in the engine number 4?"
print(question2)
answer, history = run_agent(question2, llm, history)
print(answer)

print("\n=== TEST 3: ===")
question3 = "What is z-score?"
print(question3)
answer, history = run_agent(question3, llm, history)
print(answer)

print("\n=== TEST 4: ===")
question3 = "Is engine number 5 better than engine number 4 then?"
print(question3)
answer, history = run_agent(question3, llm, history)
print(answer)



