from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.graphs.networkx_graph import KG_TRIPLE_DELIMITER


def run():
    # Prompt template for knowledge triple extraction
    _DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
        "You are a networked intelligence helping a human track knowledge triples"
        " about all relevant people, things, concepts, etc. and integrating"
        " them with your knowledge stored within your weights"
        " as well as that stored in a knowledge graph."
        " Extract all of the knowledge triples from the text."
        " A knowledge triple is a clause that contains a subject, a predicate,"
        " and an object. The subject is the entity being described,"
        " the predicate is the property of the subject that is being"
        " described, and the object is the value of the property.\n\n"
        "EXAMPLE\n"
        "It's a state in the US. It's also the number 1 producer of gold in the US.\n\n"
        f"Output: (Nevada, is a, state){KG_TRIPLE_DELIMITER}(Nevada, is in, US)"
        f"{KG_TRIPLE_DELIMITER}(Nevada, is the number 1 producer of, gold)\n"
        "END OF EXAMPLE\n\n"
        "EXAMPLE\n"
        "I'm going to the store.\n\n"
        "Output: NONE\n"
        "END OF EXAMPLE\n\n"
        "EXAMPLE\n"
        "Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.\n"
        f"Output: (Descartes, likes to drive, antique scooters){KG_TRIPLE_DELIMITER}(Descartes, plays, mandolin)\n"
        "END OF EXAMPLE\n\n"
        "EXAMPLE\n"
        "{text}"
        "Output:"
    )

    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
        input_variables=["text"],
        template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
    )

    # Instantiate the OpenAI model
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)

    # Create an LLMChain using the knowledge triple extraction prompt
    chain = LLMChain(llm=llm, prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT)

    # Run the chain with the specified text
    text = "The city of Paris is the capital and most populous city of France. The Eiffel Tower is a famous landmark in Paris."
    chain_args = {"text": text}
    triples = chain.invoke(chain_args)

    print(triples)
