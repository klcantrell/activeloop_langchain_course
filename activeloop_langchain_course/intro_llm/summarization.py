from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


def run():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    summarization_template = "Summarize the following text to one sentence: {text}"
    summarization_prompt = PromptTemplate(
        input_variables=["text"], template=summarization_template
    )
    summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)

    text = "Basketball is a team sport in which two teams, most commonly of five players each, opposing one another on a rectangular court, compete with the primary objective of shooting a basketball (approximately 9.4 inches (24 cm) in diameter) through the defender's hoop (a basket 18 inches (46 cm) in diameter mounted 10 feet (3.048 m) high to a backboard at each end of the court), while preventing the opposing team from shooting through their own hoop. A field goal is worth two points, unless made from behind the three-point line, when it is worth three. After a foul, timed play stops and the player fouled or designated to shoot a technical foul is given one, two or three one-point free throws. The team with the most points at the end of the game wins, but if regulation play expires with the score tied, an additional period of play (overtime) is mandated."
    summarized_text = summarization_chain.predict(text=text)

    print(summarized_text)
