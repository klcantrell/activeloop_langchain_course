from langchain_openai import OpenAI
from langchain.agents import load_tools, create_react_agent, AgentExecutor
from langchain import hub
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def run():
    # _with_search_example()
    _generation_only_example()


def _with_search_example():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")

    # Loading some tools to use. The llm-math tool uses an LLM, so we pass that in.
    tools = load_tools(["google-search", "llm-math"], llm=llm)

    # Create an agent and executor
    agent = create_react_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, max_iterations=6
    )

    # Testing the agent
    query = "What's the result of 1000 plus the number of goals scored in the soccer world cup in 2018?"
    print(agent_executor.invoke({"input": query}))


def _generation_only_example():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    tool_prompt = PromptTemplate(
        input_variables=["query"],
        template="You're a renowned science fiction writer. {query}",
    )
    llm_chain = LLMChain(llm=llm, prompt=tool_prompt)
    tools = [
        Tool(
            name="Science Fiction Writer",
            func=llm_chain.invoke,
            description="Use this tool for generating science fiction stories. Input should be a command about generating specific types of stories.",
        )
    ]

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")

    # Create an agent and executor
    agent = create_react_agent(llm, tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, max_iterations=6
    )

    # Testing the agent
    query = "Compose an epic science fiction saga about interstellar explorers"
    print(agent_executor.invoke({"input": query}))
