from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI


def run():
    template = """
    As a futuristic robot band conductor, I need you to help me come up with a song title.
    What's a cool song title for a song about {theme} in the year {year}?
    """
    prompt = PromptTemplate(
        input_variables=["theme", "year"],
        template=template,
    )

    # Create the LLMChain for the prompt
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

    # Input data for the prompt
    input_data = {"theme": "interstellar travel", "year": "3030"}

    # Create LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the LLMChain to get the AI-generated song title
    print(chain.invoke(input_data))
