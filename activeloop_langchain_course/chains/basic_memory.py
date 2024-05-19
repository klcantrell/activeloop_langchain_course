from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


def run():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

    conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())

    print(
        conversation.predict(
            input="List all possible words as substitute for 'artificial' as comma separated."
        )
    )
    print("\n\n\n")
    print(conversation.predict(input="And the next 4?"))
