from langchain import hub
from langchain_openai import OpenAI
from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain_google_community import GoogleSearchAPIWrapper


def run():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

    search = GoogleSearchAPIWrapper()

    tools = [
        Tool(
            name="Google Search",
            func=search.run,
            description="Useful for when you need to search Google to answer questions about current events",
        ),
    ]

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")

    # Create an agent and executor
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, max_iterations=6
    )

    response = agent_executor.invoke(
        {"input": "What's the latest news about the Mars rover?"}
    )
    print(response)
