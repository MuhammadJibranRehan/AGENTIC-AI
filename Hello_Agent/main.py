from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from env import gemini_api_key

# Reference: https://ai.google.dev/gemini-api/docs/openai
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

print("My AI can answer your questions. Type 'exit' to quit.")
print("Note: Don't provide personal or sensitive data. I don't store any information.\n")

while True:
    Question = input("Enter your question: ")
    if Question.lower() in ["exit", "quit"]:
        print("Goodbye! Thanks for using my AI assistant.")
        break

    result = Runner.run_sync(agent, Question, run_config=config)
    print("Answer:", result.final_output)
    print()
