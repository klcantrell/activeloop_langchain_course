from langchain import hub
from langchain_openai import OpenAI
from langchain.agents import create_react_agent, load_tools, AgentExecutor


def run():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    tools = load_tools(["google-search"])

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")

    # Create an agent and executor
    agent = create_react_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, max_iterations=6
    )

    print(agent_executor.invoke({"input": "What is the national drink in Spain?"}))
