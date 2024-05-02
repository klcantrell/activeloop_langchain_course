from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI


def run():
    # Initialize LLM
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

    examples = [
        {"color": "red", "emotion": "passion"},
        {"color": "blue", "emotion": "serenity"},
        {"color": "green", "emotion": "tranquility"},
    ]

    example_formatter_template = """
    Color: {color}
    Emotion: {emotion}\n
    """
    example_prompt = PromptTemplate(
        input_variables=["color", "emotion"],
        template=example_formatter_template,
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Here are some examples of colors and the emotions associated with them:\n\n",
        suffix="\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:",
        input_variables=["input"],
        example_separator="\n",
    )

    # Create the LLMChain for the prompt
    chain = LLMChain(llm=llm, prompt=few_shot_prompt)
    prompt_args = {"input": "purple"}

    # Run the LLMChain to get the AI-generated emotion associated with the input color
    response = chain.invoke(prompt_args)

    print("Color: purple")
    print("Emotion:", response)
