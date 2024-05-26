import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


def run():
    # _index_documents()

    db = _create_db(read_only=True)
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["k"] = 5

    # Uncomment the following line to apply custom filtering
    # retriever.search_kwargs['filter'] = _filter

    model = ChatOpenAI(model="gpt-4")
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

    questions = [
        # "Describe the reconciliation process. What leads to UI being updated?",
        # "What causes components to re-render?",
        # "What are hooks and how do they work?",
        # "What does the react-reconciler package do?",
        "What is renderToPipeableStream for?"
    ]
    chat_history = []

    for question in questions:
        result = qa.invoke({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")


def _filter(x):
    if "com.google" in x["text"].data()["value"]:
        return False
    metadata = x["metadata"].data()["value"]
    return "scala" in metadata["source"] or "py" in metadata["source"]


def _index_documents():
    root_dir = "activeloop_langchain_course/codebase_assistant/react"

    docs = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as _e:
                pass

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    db = _create_db()
    db.add_documents(texts)


def _create_db(read_only: bool = False):
    embeddings = OpenAIEmbeddings()

    my_activeloop_org_id = "klcantrell"
    my_activeloop_dataset_name = "langchain_course_codebase_assistant"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    return DeepLake(
        dataset_path=dataset_path, embedding=embeddings, read_only=read_only
    )
