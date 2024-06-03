import os

import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()


def run():
    # _index_documents()

    db = _create_db(read_only=True)
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["k"] = 5

    model = ChatOpenAI(model="gpt-4")
    qa = RetrievalQA.from_llm(model, retriever=retriever)

    # Set the title for the Streamlit app
    st.title("Chat with GitHub Repository")

    # Initialize the session state for placeholder messages.
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["how can I help you?"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["hello"]

    # A field input to receive user queries
    user_input = st.text_input("", key="input")

    # Search the databse and add the responses to state
    if user_input:
        output = qa.invoke(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output["result"])

    # Create the conversational UI using the previous states
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))


def _index_documents():
    root_dir = "activeloop_langchain_course/codebase_assistant_ui/tanstack-query"

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
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    my_activeloop_org_id = "klcantrell"
    my_activeloop_dataset_name = "langchain_course_codebase_assistant_ui"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    return DeepLake(
        dataset_path=dataset_path, embedding=embeddings, read_only=read_only
    )


if __name__ == "__main__":
    run()
