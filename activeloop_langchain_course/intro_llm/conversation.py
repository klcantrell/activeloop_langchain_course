from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage


def run():
    llm = ChatOpenAI(model="gpt-4")

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
        AIMessage(content="The capital of France is Paris."),
    ]

    prompt = HumanMessage(
        content="I'd like to know more about the city you just mentioned."
    )
    # add to messages
    messages.append(prompt)

    print(llm.invoke(messages))
