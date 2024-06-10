from langchain_openai import OpenAIEmbeddings, OpenAI
import faiss
from langchain_community.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain_experimental.autonomous_agents import BabyAGI


def run():
    # Define the embedding model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Initialize the vectorstore
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # set the goal
    goal = "Plan a trip to the Grand Canyon"

    # create thebabyagi agent
    # If max_iterations is None, the agent may go on forever if stuck in loops
    baby_agi = BabyAGI.from_llm(
        llm=OpenAI(model="gpt-3.5-turbo-instruct", temperature=0),
        vectorstore=vectorstore,
        verbose=False,
        max_iterations=3,
    )
    response = baby_agi({"objective": goal})
    print(response)
