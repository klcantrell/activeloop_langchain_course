from langchain_core.prompts.chat import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI


def run():
    prompt = PromptTemplate(
        template="Question: {question}\nAnswer:", input_variables=["question"]
    )

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)

    print(chain.invoke("what is the meaning of life?"))
