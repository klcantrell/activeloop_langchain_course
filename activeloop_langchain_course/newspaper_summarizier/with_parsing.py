from langchain_openai import OpenAI
from newspaper import Article
from langchain.output_parsers import PydanticOutputParser
from pydantic import field_validator, BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate


# create output parser class
class ArticleSummary(BaseModel):
    title: str = Field(description="Title of the article")
    summary: List[str] = Field(description="Bulleted list summary of the article")

    # validating whether the generated summary has at least three lines
    @field_validator("summary")
    def has_three_or_more_lines(cls, list_of_lines):
        if len(list_of_lines) < 3:
            raise ValueError("Generated summary has less than three bullet points!")
        return list_of_lines


def run():
    article_url = "https://www.artificialintelligence-news.com/2022/01/25/meta-claims-new-ai-supercomputer-will-set-records/"
    article = Article(article_url)
    article.download()
    article.parse()

    # set up output parser
    parser = PydanticOutputParser(pydantic_object=ArticleSummary)

    # create prompt template
    # notice that we are specifying the "partial_variables" parameter
    template = """
    You are a very good assistant that summarizes online articles.

    Here's the article you want to summarize.

    ==================
    Title: {article_title}

    {article_text}
    ==================

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["article_title", "article_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Format the prompt using the article title and text obtained from scraping
    formatted_prompt = prompt.format_prompt(
        article_title=article.title, article_text=article.text
    )

    # load the model
    chat = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

    # generate summary
    output = chat.invoke(formatted_prompt.to_string())
    print(parser.parse(output).summary)
