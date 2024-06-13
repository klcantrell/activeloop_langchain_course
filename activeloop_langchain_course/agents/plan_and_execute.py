from langchain_community.vectorstores import DeepLake
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import tool, Tool
from langchain.prompts import PromptTemplate
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain.chains import LLMChain
import newspaper

documents = [
    "https://www.artificialintelligence-news.com/2023/05/23/meta-open-source-speech-ai-models-support-over-1100-languages/",
    "https://www.artificialintelligence-news.com/2023/05/18/beijing-launches-campaign-against-ai-generated-misinformation/"
    "https://www.artificialintelligence-news.com/2023/05/16/openai-ceo-ai-regulation-is-essential/",
    "https://www.artificialintelligence-news.com/2023/05/15/jay-migliaccio-ibm-watson-on-leveraging-ai-to-improve-productivity/",
    "https://www.artificialintelligence-news.com/2023/05/15/iurii-milovanov-softserve-how-ai-ml-is-helping-boost-innovation-and-personalisation/",
    "https://www.artificialintelligence-news.com/2023/05/11/ai-and-big-data-expo-north-america-begins-in-less-than-one-week/",
    "https://www.artificialintelligence-news.com/2023/05/11/eu-committees-green-light-ai-act/",
    "https://www.artificialintelligence-news.com/2023/05/09/wozniak-warns-ai-will-power-next-gen-scams/",
    "https://www.artificialintelligence-news.com/2023/05/09/infocepts-ceo-shashank-garg-on-the-da-market-shifts-and-impact-of-ai-on-data-analytics/",
    "https://www.artificialintelligence-news.com/2023/05/02/ai-godfather-warns-dangers-and-quits-google/",
    "https://www.artificialintelligence-news.com/2023/04/28/palantir-demos-how-ai-can-used-military/",
    "https://www.artificialintelligence-news.com/2023/04/26/ftc-chairwoman-no-ai-exemption-to-existing-laws/",
    "https://www.artificialintelligence-news.com/2023/04/24/bill-gates-ai-teaching-kids-literacy-within-18-months/",
    "https://www.artificialintelligence-news.com/2023/04/21/google-creates-new-ai-division-to-challenge-openai/",
]


def run():
    # _build_index()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Write a summary of the following text: {query}",
    )
    summarize_chain = LLMChain(llm=llm, prompt=prompt)

    # let's create the Plan and Execute agent
    planner = load_chat_planner(llm)
    executor = load_agent_executor(
        llm,
        [
            retrieve_n_docs_tool,
            Tool(
                name="Summarizer",
                func=summarize_chain.invoke,
                description="Useful for summarizing texts",
            ),
        ],
        verbose=True,
    )
    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

    # we test the agent
    response = agent.invoke(
        "Write an overview of Artificial Intelligence regulations by governments of different countries. Summarize and compare the regulations set by each country. Stop researching once you have enough information for an overview. Don't worry about remembering the original question, you can provide your summary once it's complete."
    )
    print(response)


def _build_index():
    db = _create_db()

    # fetch documents
    pages_content = []
    for url in documents:
        try:
            article = newspaper.Article(url)
            article.download()
            article.parse()
            if len(article.text) > 0:
                pages_content.append({"url": url, "text": article.text})
        except Exception as e:
            print(f"Error occurred while fetching article at {url}: {e}")

    # split up documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_texts = []
    for d in pages_content:
        chunks = text_splitter.split_text(d["text"])
        for chunk in chunks:
            all_texts.append(chunk)

    # add them to DeepLake
    db.add_texts(all_texts)


def _create_db(read_only: bool = False):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    my_activeloop_org_id = "klcantrell"
    my_activeloop_dataset_name = "langchain_course_plan_and_execute_example"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    return DeepLake(
        dataset_path=dataset_path, embedding=embeddings, read_only=read_only
    )


# We define some variables that will be used inside our custom tool
CUSTOM_TOOL_DOCS_SEPARATOR = "\n---------------\n"  # how to join together the retrieved docs to form a single string


# We use the tool decorator to wrap a function that will become our custom tool
# Note that the tool has a single string as input and returns a single string as output
# The name of the function will be the name of our custom tool
# The docstring of the function will be the description of our custom tool
# The description is used by the agent to decide whether to use the tool for a specific query
@tool
def retrieve_n_docs_tool(query: str) -> str:
    """Searches for relevant documents that may contain the answer to the query."""

    db = _create_db(read_only=True)
    retriever = db.as_retriever()
    retriever.search_kwargs["k"] = 3
    docs = retriever.invoke(query)
    texts = [doc.page_content for doc in docs]
    texts_merged = (
        "---------------\n"
        + CUSTOM_TOOL_DOCS_SEPARATOR.join(texts)
        + "\n---------------"
    )
    return texts_merged
