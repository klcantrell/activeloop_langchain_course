from typing import List
from pydantic import BaseModel, Field, field_validator
from langchain_openai import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


# Define your desired data structure.
class Suggestions(BaseModel):
    words: List[str] = Field(description="list of substitue words based on context")
    reasons: List[str] = Field(
        description="the reasoning of why this word fits the context"
    )

    # Throw error in case of receiving a numbered-list from API
    @field_validator("words")
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word can not start with numbers!")
        return field

    @field_validator("reasons")
    def end_with_dot(cls, field):
        for idx, item in enumerate(field):
            if item[-1] != ".":
                field[idx] += "."
        return field


def run():
    parser = PydanticOutputParser(pydantic_object=Suggestions)

    template = """
    Offer a list of suggestions to substitute the specified target_word based on the presented context and the reason each word was selected. Each word should get its own reason. For example, if "mighty" is selected as a word, then the reason should begin with "mighty was chosen because...".
    {format_instructions}
    target_word={target_word}
    context={context}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["target_word", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    model_input = prompt.format_prompt(
        target_word="behaviour",
        context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson.",
    )

    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.0)

    output = llm.invoke(model_input.to_string())

    print(parser.parse(output))
