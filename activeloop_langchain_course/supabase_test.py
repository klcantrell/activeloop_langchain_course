import os

from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.agents import create_react_agent, Tool, AgentExecutor


def run():
    # _index_documents()

    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    vector_store = _create_db()
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
    )

    tools = [
        Tool(
            name="Retrieval QA System",
            func=retrieval_qa.invoke,
            description="Useful for answering questions.",
        ),
    ]

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")

    # Create an agent and executor
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": "When was Ka Klam born?"})
    print(response)


def _index_documents():
    texts = [
        "Napoleon Bonaparte was born in 15 August 1769",
        "Louis XIV was born in 5 September 1638",
        "Lady Gaga was born in 28 March 1986",
        "Michael Jeffrey Jordan was born in 17 February 1963",
        "Ka Klam was born on 20 February 1989",
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts_split = [
        split for splits in map(text_splitter.split_text, texts) for split in splits
    ]

    vector_store = _create_db()
    vector_store.add_texts(
        texts=texts_split, ids=[i + 1 for i, _ in enumerate(texts_split)]
    )


def _create_db():
    embeddings = OpenAIEmbeddings()

    # Load environment variables from .env file and return the keys
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    supabase_client = create_client(supabase_url, supabase_key)
    return SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )
