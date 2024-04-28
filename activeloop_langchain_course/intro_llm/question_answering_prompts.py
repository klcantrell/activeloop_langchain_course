from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain.chains import LLMChain


def run():
    template = """Question: {question}

    Answer: """
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # user question
    question = "What is the capital city of France?"

    # initialize Hub LLM
    hub_llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0},
    )

    # create prompt template > LLM chain
    llm_chain = LLMChain(prompt=prompt, llm=hub_llm)

    # ask the user question about the capital of France
    print(llm_chain.invoke(question))

    # generate multiple respones to multiple questions
    questions = [
        {"question": "What is the capital city of France?"},
        {"question": "What is the largest mammal on Earth?"},
        {"question": "Which gas is most abundant in Earth's atmosphere?"},
        {"question": "What color is a ripe banana?"},
    ]
    print(llm_chain.generate(questions))

    # get multiple questions answered in single prompt. works with more capable models.
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
    multi_template = """Answer the following questions one at a time.

    Questions:
    {questions}

    Answers:
    """
    long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

    llm_chain = LLMChain(prompt=long_prompt, llm=llm)

    qs_str = (
        "What is the capital city of France?\n"
        + "What is the largest mammal on Earth?\n"
        + "Which gas is most abundant in Earth's atmosphere?\n"
        + "What color is a ripe banana?\n"
    )
    print(llm_chain.invoke(qs_str))
