from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain


def run():
    prompt_template = "What is a word to replace the following: {word}?"

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

    # simple usage
    print(llm_chain.invoke("artificial"))

    # usage with a list of inputs
    input_list = [{"word": "artificial"}, {"word": "intelligence"}, {"word": "robot"}]
    print(llm_chain.apply(input_list))

    # usage with generate
    print(llm_chain.generate(input_list))

    # usage with predict
    prompt_template = "Looking at the context of '{context}'. What is an appropriate word to replace the following: {word}?"

    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=prompt_template, input_variables=["word", "context"]
        ),
    )

    print(llm_chain.predict(word="fan", context="inanimate objects"))
    print(llm_chain.predict(word="fan", context="humans"))
