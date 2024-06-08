from langchain import hub
from langchain_openai import OpenAI
from langchain.agents import create_react_agent, load_tools, Tool, AgentExecutor
from langchain_experimental.utilities import PythonREPL


def run():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    tools = load_tools(["wikipedia"])

    python_repl = PythonREPL()
    tools.append(
        Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,
        )
    )

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")

    # Create an agent and executor
    agent = create_react_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    print(
        agent_executor.invoke(
            {
                "input": "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?"
            }
        )
    )
