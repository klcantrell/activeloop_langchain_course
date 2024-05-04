from typing import List
from pydantic import BaseModel, Field, field_validator
from langchain_openai import OpenAI
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser


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
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=Suggestions)

    missformatted_output = '{"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}'

    outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    print(outputfixing_parser.parse(missformatted_output))
