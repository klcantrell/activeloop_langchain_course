from langchain_community.vectorstores import DeepLake
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings


def run():
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # create Deep Lake dataset
    my_activeloop_org_id = "klcantrell"
    my_activeloop_dataset_name = "langchain_course_using_selectors"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

    # Examples of a pretend task of creating antonyms.
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "energetic", "output": "lethargic"},
        {"input": "sunny", "output": "gloomy"},
        {"input": "windy", "output": "calm"},
    ]

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        # The list of examples available to select from.
        examples,
        # The embedding class used to produce embeddings which are used to measure semantic similarity.
        embeddings,
        # The VectorStore class that is used to store the embeddings and do a similarity search over.
        db,
        # The number of examples to produce.
        k=1,
    )
    similar_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input",
        suffix="Input: {adjective}\nOutput:",
        input_variables=["adjective"],
    )

    # Input is a feeling, so should select the happy/sad example
    print(similar_prompt.format(adjective="worried"))

    # Input is a measurement, so should select the tall/short example
    print(similar_prompt.format(adjective="large"))

    # You can add new examples to the SemanticSimilarityExampleSelector as well
    similar_prompt.example_selector.add_example(
        {"input": "enthusiastic", "output": "apathetic"}
    )
    print(similar_prompt.format(adjective="passionate"))
