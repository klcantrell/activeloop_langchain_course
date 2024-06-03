from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate




def run():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
        MessagesPlaceholder(variable_name="history"), # can be replaced with history if the prompt needs to be customized
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

    output = conversation.predict(input="Hi there!")
    output = conversation.predict(
        input="In what scenarios extra memory should be used?"
    )
    output = conversation.predict(
        input="There are various types of memory in Langchain. When to use which type?"
    )
    output = conversation.predict(input="Do you remember what was our first message?")

    print(output)
