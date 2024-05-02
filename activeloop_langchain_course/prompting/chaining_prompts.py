from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI


def run():
    # Initialize LLM
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

    # Prompt 1
    template_question = """What is the name of the famous scientist who developed the theory of general relativity?
    Answer: """
    prompt_question = PromptTemplate(template=template_question, input_variables=[])

    # Create the LLMChain for the first prompt
    chain_question = LLMChain(llm=llm, prompt=prompt_question)

    # Run the LLMChain for the first prompt with an empty dictionary
    response_question = chain_question.invoke({})

    # Extract the scientist's name from the response
    scientist = response_question["text"].strip()

    # Prompt 2
    template_fact = """Provide a brief description of {scientist}'s theory of general relativity.
    Answer: """
    prompt_fact = PromptTemplate(input_variables=["scientist"], template=template_fact)

    # Create the LLMChain for the second prompt
    chain_fact = LLMChain(llm=llm, prompt=prompt_fact)

    # Input data for the second prompt
    input_data = {"scientist": scientist}

    # Run the LLMChain for the second prompt
    print(chain_fact.invoke(input_data))
