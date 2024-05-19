from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser


def run():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    output_parser = CommaSeparatedListOutputParser()

    template = """List all possible words as substitute for 'artificial' as comma separated.

    Current conversation:
    {history}

    {input}"""

    conversation = ConversationChain(
        llm=llm,
        prompt=PromptTemplate(
            template=template,
            input_variables=["history", "input"],
            output_parser=output_parser,
        ),
        memory=ConversationBufferMemory(),
        verbose=True,
    )

    print(conversation.predict(input=""))
