from langchain import hub
from langchain_openai import OpenAI
from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain_experimental.utilities import PythonREPL


def run():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    python_repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")

    # Create an agent and executor
    agent = create_react_agent(llm, [repl_tool], prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[repl_tool],
        verbose=True,
        handle_parsing_errors=True,
    )

    print(agent_executor.invoke({"input": "What is 20 to the power of 0.23?"}))
