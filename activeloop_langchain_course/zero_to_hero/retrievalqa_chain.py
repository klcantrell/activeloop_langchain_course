from langchain import hub
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.agents import create_react_agent, Tool, AgentExecutor


def run():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # create Deep Lake dataset
    my_activeloop_org_id = "klcantrell"
    my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=db.as_retriever()
    )

    tools = [
        Tool(
            name="Retrieval QA System",
            func=retrieval_qa.invoke,
            description="Useful for answering questions.",
        ),
    ]

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")

    # Create an agent and executor
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": "When was Michael Jordan born?"})
    print(response)
