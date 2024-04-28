from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def run():
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    template = "You are an assistant that helps users find information about movies."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Find information about the movie {movie_title}."
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    response = chat.invoke(
        chat_prompt.format_prompt(movie_title="Inception").to_messages()
    )

    print(response.content)
