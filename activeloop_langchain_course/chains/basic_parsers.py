from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain


def run():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

    output_parser = CommaSeparatedListOutputParser()
    template = (
        """List all possible words as substitute for 'artificial' as comma separated."""
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=template, output_parser=output_parser, input_variables=[]
        ),
        output_parser=output_parser,
    )

    print(llm_chain.predict())
