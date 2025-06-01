from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

gemini_api_key = "AIzaSyBZ2BjGDntOJZKizMUd3M4_3QETiyiVytM"

Question = input("Enter your question: ")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant")
print("My AI can answer your questions. Dont provide personal information or sensitive data. I dont store any infromation")

result = Runner.run_sync(agent, Question, run_config=config)

print(result.final_output)