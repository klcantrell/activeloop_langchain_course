from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple


def run():
    evil_assistant_prompt = PromptTemplate(
        template="""
                You are a evil mentor for students with no morals. Give suggestions that are easiest and fastest to achieve the goal.
                Goal: {inquiry}
                Easiest way:""",
        input_variables=["inquiry"],
    )

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
    evil_assistant_chain = LLMChain(llm=llm, prompt=evil_assistant_prompt)

    ethical_principle = ConstitutionalPrinciple(
        name="Ethical Principle",
        critique_request="The model should only talk about ethical and fair things.",
        revision_request="Rewrite the model's output to be both ethical and fair.",
    )
    fun_principle = ConstitutionalPrinciple(
        name="Be Funny",
        critique_request="The model responses must be funny and understandable for a 7th grader.",
        revision_request="Rewrite the model's output to be both funny and understandable for 7th graders.",
    )

    constitutional_chain = ConstitutionalChain.from_llm(
        chain=evil_assistant_chain,
        constitutional_principles=[ethical_principle, fun_principle],
        llm=llm,
        verbose=True,
    )

    result = constitutional_chain.invoke(
        input={"inquiry": "Getting full mark on my exams."}
    )

    print(result)
