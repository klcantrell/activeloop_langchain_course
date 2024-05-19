from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain


def run():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

    # poet
    poet_template = """You are an American poet, your job is to come up with\
    poems based on a given theme.

    Here is the theme you have been asked to generate a poem on:
    {input}\
    """

    poet_prompt_template = PromptTemplate(
        input_variables=["input"], template=poet_template
    )

    # creating the poet chain
    poet_chain = LLMChain(llm=llm, output_key="poem", prompt=poet_prompt_template)

    # critic
    critic_template = """You are a critic of poems, you are tasked\
    to inspect the themes of poems. Identify whether the poem includes romantic expressions or descriptions of nature.

    Your response should be in the following format, as a Python Dictionary.
    poem: this should be the poem you received 
    Romantic_expressions: True or False
    Nature_descriptions: True or False

    Here is the poem submitted to you:
    {poem}\
    """

    critic_prompt_template = PromptTemplate(
        input_variables=["poem"], template=critic_template
    )

    # creating the critic hain
    critic_chain = LLMChain(
        llm=llm, output_key="critic_verified", prompt=critic_prompt_template
    )

    overall_chain = SimpleSequentialChain(chains=[poet_chain, critic_chain])

    # Run the poet and critic chain with a specific theme
    theme = "the beauty of nature"
    review = overall_chain.invoke(theme)

    # Print the review to see the critic's evaluation
    print(review["output"])
