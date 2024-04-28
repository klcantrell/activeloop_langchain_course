from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback


def run():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", n=2, best_of=2)

    with get_openai_callback() as cb:
        _result = llm.invoke("Tell me a joke")
        print(cb)
