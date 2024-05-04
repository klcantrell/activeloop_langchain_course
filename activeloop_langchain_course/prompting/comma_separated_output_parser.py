from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser


def run():
    parser = CommaSeparatedListOutputParser()

    # Prepare the Prompt
    template = """
    Offer a list of suggestions to substitute the word '{target_word}' based the presented the following text: {context}. Limit your suggestions to 5 words.
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["target_word", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model_input = prompt.format(
        target_word="behaviour",
        context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson.",
    )

    # Loading OpenAI API
    model = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.0)

    # Send the Request
    output = model.invoke(model_input)
    print(parser.parse(output))
