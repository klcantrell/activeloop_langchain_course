from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


def run():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
    translation_prompt = PromptTemplate(
        input_variables=["source_language", "target_language", "text"],
        template=translation_template,
    )
    translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

    source_language = "English"
    target_language = "French"
    text = "Your text here"
    translated_text = translation_chain.predict(
        source_language=source_language, target_language=target_language, text=text
    )

    print(translated_text)
