from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_community.tools import WriteFileTool, ReadFileTool
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.agents import Tool
from langchain_google_community import GoogleSearchAPIWrapper


def run():
    # Define the embedding model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    search = GoogleSearchAPIWrapper()
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="Useful for when you need to answer questions about current events. You should ask targeted questions",
            return_direct=True,
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]

    # Initialize the vectorstore
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # Set up the model and AutoGPT
    agent = AutoGPT.from_llm_and_tools(
        ai_name="Jim",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        memory=vectorstore.as_retriever(),
    )

    # Set verbose to be true
    agent.chain.verbose = True

    task = "Provide an analysis of the major historical events that led to the French Revolution"

    agent.run([task])
